#!/usr/bin/env python3
"""
🎯 Interactive Mesh Segmentation with Gradio
Complete control over SAM2/SAM3 parameters for mesh segmentation
"""

import gradio as gr
import trimesh
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from omegaconf import OmegaConf
import sys
import os
import warnings

# Suppress MobileSAM registry overwrite UserWarnings (conflict with timm/other vision libs)
warnings.filterwarnings(
    "ignore",
    message="Overwriting .* in registry",
    category=UserWarning,
)

# PyTorch 2.6+ weights_only=True: patch torch.load to use weights_only=False for trusted checkpoints
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to default to weights_only=False for Ultralytics/FastSAM checkpoints"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add the samesh package to the path (repo root / src)
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from samesh.models.sam_mesh import segment_mesh
from samesh.data.loaders import read_mesh

# Helper functions (from the notebook)
def get_first_mesh_or_combined(mesh_or_scene) -> trimesh.Trimesh:
    """Extract single mesh from Trimesh or Scene object"""
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    elif isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            raise ValueError("No geometry found in the scene.")
        combined = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
        return combined
    else:
        raise TypeError("Loaded object is neither a Trimesh nor a Scene.")

def merge_vertices_by_distance_blender_style(mesh: trimesh.Trimesh, threshold: float = 1e-4) -> trimesh.Trimesh:
    """Merge vertices by distance using KDTree (Blender-style)"""
    mesh = mesh.copy()
    verts = mesh.vertices
    
    if len(verts) == 0:
        return mesh
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        
        # Find all pairs within threshold distance
        pairs = tree.query_pairs(threshold)
        
        if len(pairs) == 0:
            return mesh
        
        # Create vertex mapping
        vertex_map = np.arange(len(verts))
        
        # Process pairs to merge vertices
        for i, j in pairs:
            # Find root of both vertices
            root_i = i
            while vertex_map[root_i] != root_i:
                root_i = vertex_map[root_i]
            
            root_j = j
            while vertex_map[root_j] != root_j:
                root_j = vertex_map[root_j]
            
            # Merge to smaller index
            if root_i != root_j:
                if root_i < root_j:
                    vertex_map[root_j] = root_i
                else:
                    vertex_map[root_i] = root_j
        
        # Compress paths and get final mapping
        for i in range(len(vertex_map)):
            root = i
            while vertex_map[root] != root:
                root = vertex_map[root]
            vertex_map[i] = root
        
        # Get unique vertices and create new mapping
        unique_indices = np.unique(vertex_map)
        new_vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        # Create new vertices and faces
        new_vertices = verts[unique_indices]
        new_faces = np.array([[new_vertex_map[vertex_map[face[0]]], 
                              new_vertex_map[vertex_map[face[1]]], 
                              new_vertex_map[vertex_map[face[2]]]] 
                             for face in mesh.faces])
        
        # Create new mesh
        new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # Copy other attributes if they exist
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            new_mesh.visual.face_colors = mesh.visual.face_colors
        
        return new_mesh
        
    except ImportError:
        # Fallback to trimesh's built-in merge_vertices if scipy not available
        mesh.merge_vertices()
        return mesh

# Global variables
current_mesh = None
segmented_mesh = None

def load_mesh_file(file_path):
    """Load and preprocess a mesh file"""
    global current_mesh
    
    if file_path is None:
        return "❌ No file selected", None
    
    try:
        # Load the mesh
        mesh = trimesh.load(file_path)
        current_mesh = get_first_mesh_or_combined(mesh)
        
        # Apply vertex merging for better segmentation
        current_mesh = merge_vertices_by_distance_blender_style(current_mesh)
        
        info = f"""
✅ **Mesh Loaded Successfully!**
- **Vertices:** {len(current_mesh.vertices):,}
- **Faces:** {len(current_mesh.faces):,}
- **Bounds:** {current_mesh.bounds}
- **Volume:** {current_mesh.volume:.2f}
- **Surface Area:** {current_mesh.area:.2f}
"""
        
        # Save a temporary preview
        preview_path = "/tmp/mesh_preview.glb"
        current_mesh.export(preview_path)
        
        return info, preview_path
        
    except Exception as e:
        return f"❌ Error loading mesh: {str(e)}", None

def create_config(
    # Model selection
    model_type,
    
    # SAM2 parameters
    sam2_points_per_side,
    sam2_pred_iou_thresh,
    sam2_stability_score_thresh,
    
    # SAM3 parameters  
    use_smart_prompts,
    use_grid_prior,
    gemini_api_key,
    sam3_text_prompt,
    sam3_threshold,
    sam3_mask_threshold,
    sam3_points_per_side,
    
    # SAM Original parameters
    sam_points_per_side,
    sam_pred_iou_thresh,
    sam_stability_score_thresh,
    
    # SAM-HQ parameters
    sam_hq_points_per_side,
    sam_hq_pred_iou_thresh,
    sam_hq_stability_score_thresh,
    
    # FastSAM parameters
    fastsam_checkpoint,
    fastsam_conf_thresh,
    fastsam_iou_thresh,
    fastsam_imgsz,
    
    # MobileSAM parameters
    mobilesam_points_per_side,
    mobilesam_pred_iou_thresh,
    mobilesam_stability_score_thresh,
    
    # Mesh processing parameters
    use_modes,
    min_area,
    connections_bin_resolution,
    connections_bin_threshold_percentage,
    smoothing_threshold_percentage_size,
    smoothing_threshold_percentage_area,
    smoothing_iterations,
    repartition_lambda,
    repartition_iterations,
    
    # Renderer parameters
    target_dim,
    sampling_radius
):
    """Create configuration dictionary from parameters"""
    
    config = {
        'cache': '/tmp/samesh_cache',  # override with config or env if needed
        'cache_overwrite': True,
        'output': '/tmp/samesh_output',
        
        'sam': {
            'model_type': model_type,
        },
    }
    
    # Add model-specific configuration based on selected model type
    if model_type == 'sam2':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt'),
            'model_config': 'sam2_hiera_l.yaml',
            'auto': True,
            'ground': True,
            'engine_config': {
                'points_per_side': sam2_points_per_side,
                'crop_n_layers': 0,
                'pred_iou_thresh': sam2_pred_iou_thresh,
                'stability_score_thresh': sam2_stability_score_thresh,
                'stability_score_offset': 1.0
            }
        }
    elif model_type == 'sam3':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / 'sam3.pt'),
            'auto': True,
            'use_smart_prompts': use_smart_prompts,
            'use_grid_prior': use_grid_prior,
            'gemini_api_key': gemini_api_key if gemini_api_key.strip() else None,
            'text_prompt': sam3_text_prompt if not use_smart_prompts else None,
            'threshold': sam3_threshold,
            'mask_threshold': sam3_mask_threshold,
            'engine_config': {
                'points_per_side': sam3_points_per_side
            }
        }
    elif model_type == 'sam':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth'),
            'auto': True,
            'ground': False,
            'engine_config': {
                'points_per_side': sam_points_per_side,
                'pred_iou_thresh': sam_pred_iou_thresh,
                'stability_score_thresh': sam_stability_score_thresh,
                'crop_n_layers': 0,
                'stability_score_offset': 1.0
            }
        }
    elif model_type == 'sam_hq':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / 'sam_hq_vit_h.pth'),
            'auto': True,
            'ground': False,
            'engine_config': {
                'points_per_side': sam_hq_points_per_side,
                'pred_iou_thresh': sam_hq_pred_iou_thresh,
                'stability_score_thresh': sam_hq_stability_score_thresh,
                'crop_n_layers': 0,
                'stability_score_offset': 1.0
            }
        }
    elif model_type == 'fastsam':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / fastsam_checkpoint),
            'auto': True,
            'engine_config': {
                'points_per_side': 32,  # Not used by FastSAM, but required by config schema
                'conf': fastsam_conf_thresh,
                'iou': fastsam_iou_thresh,
                'imgsz': int(fastsam_imgsz),
            }
        }
    elif model_type == 'mobilesam':
        config['sam']['sam'] = {
            'checkpoint': str(_REPO_ROOT / 'checkpoints' / 'mobile_sam.pt'),
            'auto': True,
            'ground': False,
            'engine_config': {
                'points_per_side': mobilesam_points_per_side,
                'pred_iou_thresh': mobilesam_pred_iou_thresh,
                'stability_score_thresh': mobilesam_stability_score_thresh,
                'crop_n_layers': 0,
                'stability_score_offset': 1.0
            }
        }
    
    config.update({
        'sam_mesh': {
            'use_modes': use_modes,
            'min_area': min_area,
            'connections_bin_resolution': connections_bin_resolution,
            'connections_bin_threshold_percentage': connections_bin_threshold_percentage,
            'smoothing_threshold_percentage_size': smoothing_threshold_percentage_size,
            'smoothing_threshold_percentage_area': smoothing_threshold_percentage_area,
            'smoothing_iterations': smoothing_iterations,
            'repartition_cost': 1,
            'repartition_lambda': repartition_lambda,
            'repartition_iterations': repartition_iterations
        },
        
        'renderer': {
            'target_dim': [target_dim, target_dim],
            'camera_generation_method': 'icosahedron',
            'renderer_args': {
                'interpolate_norms': True
            },
            'sampling_args': {'radius': sampling_radius},
            'lighting_args': {}
        }
    })
    
    # Model-specific config is already set above, no need to update
    if model_type == 'sam3':
        # Ensure grid prior setting is passed through
        config['sam']['sam']['use_grid_prior'] = use_grid_prior
    
    return config

def segment_mesh_with_params(
    # Model selection
    model_type,
    
    # SAM2 parameters
    sam2_points_per_side,
    sam2_pred_iou_thresh,
    sam2_stability_score_thresh,
    
    # SAM3 parameters  
    use_smart_prompts,
    use_grid_prior,
    gemini_api_key,
    sam3_text_prompt,
    sam3_threshold,
    sam3_mask_threshold,
    sam3_points_per_side,
    
    # SAM Original parameters
    sam_points_per_side,
    sam_pred_iou_thresh,
    sam_stability_score_thresh,
    
    # SAM-HQ parameters
    sam_hq_points_per_side,
    sam_hq_pred_iou_thresh,
    sam_hq_stability_score_thresh,
    
    # FastSAM parameters
    fastsam_checkpoint,
    fastsam_conf_thresh,
    fastsam_iou_thresh,
    fastsam_imgsz,
    
    # MobileSAM parameters
    mobilesam_points_per_side,
    mobilesam_pred_iou_thresh,
    mobilesam_stability_score_thresh,
    
    # Mesh processing parameters
    use_modes,
    min_area,
    connections_bin_resolution,
    connections_bin_threshold_percentage,
    smoothing_threshold_percentage_size,
    smoothing_threshold_percentage_area,
    smoothing_iterations,
    repartition_lambda,
    repartition_iterations,
    
    # Renderer parameters
    target_dim,
    sampling_radius,
    
    # Progress callback
    progress=gr.Progress()
):
    """Run mesh segmentation with the given parameters"""
    global current_mesh, segmented_mesh
    
    if current_mesh is None:
        return "❌ Please load a mesh first!", None, None
    
    try:
        progress(0.1, desc="Creating configuration...")
        
        # Create config
        config = create_config(
            model_type, sam2_points_per_side, sam2_pred_iou_thresh, sam2_stability_score_thresh,
            use_smart_prompts, use_grid_prior, gemini_api_key, sam3_text_prompt, sam3_threshold, sam3_mask_threshold, sam3_points_per_side,
            sam_points_per_side, sam_pred_iou_thresh, sam_stability_score_thresh,
            sam_hq_points_per_side, sam_hq_pred_iou_thresh, sam_hq_stability_score_thresh,
            fastsam_checkpoint, fastsam_conf_thresh, fastsam_iou_thresh, fastsam_imgsz,
            mobilesam_points_per_side, mobilesam_pred_iou_thresh, mobilesam_stability_score_thresh,
            use_modes, min_area, connections_bin_resolution, connections_bin_threshold_percentage,
            smoothing_threshold_percentage_size, smoothing_threshold_percentage_area,
            smoothing_iterations, repartition_lambda, repartition_iterations,
            target_dim, sampling_radius
        )
        
        # Convert to OmegaConf
        config = OmegaConf.create(config)
        
        progress(0.2, desc="Initializing model...")
        
        # Create temporary mesh file
        temp_mesh_path = "/tmp/input_mesh.obj"
        current_mesh.export(temp_mesh_path)
        
        progress(0.5, desc=f"Running {model_type.upper()} segmentation...")
        
        # Run segmentation using the standalone function
        segmented_mesh = segment_mesh(temp_mesh_path, config)
        
        progress(0.8, desc="Processing results...")
        
        # Analyze results
        if hasattr(segmented_mesh.visual, 'face_colors'):
            face_colors = segmented_mesh.visual.face_colors
            if face_colors is not None:
                unique_colors = np.unique(face_colors.reshape(-1, face_colors.shape[-1]), axis=0)
                num_segments = len(unique_colors)
            else:
                num_segments = 1
        else:
            num_segments = 1
        
        # Save result
        output_path = "/tmp/segmented_mesh.glb"
        segmented_mesh.export(output_path)
        
        progress(1.0, desc="Complete!")
        
        result_info = f"""
✅ **Segmentation Complete!**
- **Model Used:** {model_type.upper()}
- **Segments Found:** {num_segments}
- **Original Faces:** {len(current_mesh.faces):,}
- **Segmented Faces:** {len(segmented_mesh.faces):,}
- **Processing Mode:** {use_modes}
"""

        if model_type == 'sam3':
            if use_smart_prompts:
                result_info += f"- **Smart Prompts:** Enabled (AI-generated)\n"
            else:
                result_info += f"- **Manual Prompt:** '{sam3_text_prompt}'\n"
            
            if use_grid_prior:
                result_info += f"- **Grid Priors:** Enabled (Hybrid SAM2+SAM3)"
            else:
                result_info += f"- **Grid Priors:** Disabled (Text-only)"
        
        return result_info, output_path, output_path
        
    except Exception as e:
        error_msg = f"❌ **Segmentation Failed**\n\nError: {str(e)}\n\nTry adjusting parameters or switching models."
        return error_msg, None, None

def get_preset_configs():
    """Get preset configurations for quick testing"""
    presets = {
        "Conservative (Few Segments)": {
            "sam3_threshold": 0.7,
            "sam3_mask_threshold": 0.7,
            "min_area": 1000,
            "repartition_lambda": 5,
            "smoothing_iterations": 50
        },
        "Balanced": {
            "sam3_threshold": 0.3,
            "sam3_mask_threshold": 0.3,
            "min_area": 256,
            "repartition_lambda": 3,
            "smoothing_iterations": 32
        },
        "Aggressive (Many Segments)": {
            "sam3_threshold": 0.1,
            "sam3_mask_threshold": 0.1,
            "min_area": 64,
            "repartition_lambda": 1,
            "smoothing_iterations": 5
        },
        "Ultra-Aggressive": {
            "sam3_threshold": 0.05,
            "sam3_mask_threshold": 0.05,
            "min_area": 16,
            "repartition_lambda": 1,
            "smoothing_iterations": 1
        }
    }
    return presets

def apply_preset(preset_name):
    """Apply a preset configuration"""
    presets = get_preset_configs()
    if preset_name in presets:
        return list(presets[preset_name].values())
    return [0.3, 0.3, 256, 3, 32]  # Default values

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="🎯 Interactive Mesh Segmentation") as demo:
        
        gr.Markdown("**🎯 Interactive Mesh Segmentation** — Upload a mesh, pick a SAM model, then Run. Results appear in the right column.")

        with gr.Row():
            mesh_file = gr.File(
                label="📁 Upload Mesh File",
                file_types=[".glb", ".obj", ".stl", ".ply"],
                type="filepath",
                scale=1
            )

        with gr.Row():
            mesh_info = gr.Textbox(
                label="Mesh Information",
                lines=3,
                max_lines=5,
                interactive=False,
                scale=1,
                min_width=180
            )
            mesh_preview = gr.Model3D(
                label="Mesh Preview",
                height=180,
                scale=1,
                min_width=200
            )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### ⚙️ Parameters")

                MODEL_CHOICES = [
                    ("SAM3", "sam3"),
                    ("SAM2", "sam2"),
                    ("FastSAM", "fastsam"),
                    ("SAM (original)", "sam"),
                    ("SAM-HQ", "sam_hq"),
                    ("MobileSAM", "mobilesam"),
                ]
                model_type = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="sam3",
                    label="Model",
                    info="SAM2/SAM3 work out of the box. FastSAM optional: pip install fastsam (then download FastSAM-x.pt checkpoint).",
                    allow_custom_value=False,
                )
                preset_dropdown = gr.Dropdown(
                    choices=list(get_preset_configs().keys()),
                    value="Balanced",
                    label="Preset",
                    info="Quick apply: Conservative, Balanced, Aggressive, Ultra-Aggressive"
                )

                MODEL_DESCRIPTIONS = {
                    "sam": "**SAM (original)** — Meta 2023. First Segment Anything Model, ViT-based. Legacy; use SAM2 for better quality and speed.",
                    "sam2": "**SAM2** — Meta 2024. Direct successor of SAM. Hiera backbone, faster and better masks. Best default for mesh segmentation.",
                    "sam3": "**SAM3** — Meta, latest. Adds *text* prompting (e.g. \"chair leg\", \"surface\"). Optional Gemini agent for smart prompts.",
                    "sam_hq": "**SAM-HQ** — Community (SysCV). Same as SAM with sharper mask boundaries; high-quality output token. Good for fine edges.",
                    "fastsam": "**FastSAM** — YOLOv8-based. ~50× faster, automatic \"everything\" masking. Best when speed matters.",
                    "mobilesam": "**MobileSAM** — Lightweight ViT-tiny. Small and fast for edge/mobile; fewer parameters than full SAM.",
                }
                model_description = gr.Markdown(value=MODEL_DESCRIPTIONS["sam3"])

                with gr.Accordion("Model parameters", open=True):
                    with gr.Row(visible=False) as row_sam2:
                        with gr.Column():
                            sam2_points_per_side = gr.Slider(8, 128, value=32, step=8, label="Points Per Side")
                            sam2_pred_iou_thresh = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Prediction IoU Threshold")
                            sam2_stability_score_thresh = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Stability Score Threshold")

                    with gr.Row(visible=True) as row_sam3:
                        with gr.Column():
                            use_smart_prompts = gr.Checkbox(value=True, label="🧠 Use Gemini 2.5 Flash Intelligent Agent")
                            use_grid_prior = gr.Checkbox(value=True, label="🔲 Use Grid-Based Priors (Hybrid SAM2+SAM3)")
                            gemini_api_key = gr.Textbox(value="", label="Google AI API Key (optional)", type="password")
                            sam3_text_prompt = gr.Textbox(value="individual part", label="Manual Text Prompt")
                            sam3_threshold = gr.Slider(0.01, 1.0, value=0.3, step=0.01, label="Segmentation Threshold")
                            sam3_mask_threshold = gr.Slider(0.01, 1.0, value=0.3, step=0.01, label="Mask Threshold")
                            sam3_points_per_side = gr.Slider(8, 256, value=64, step=8, label="Points Per Side")

                    with gr.Row(visible=False) as row_sam:
                        with gr.Column():
                            sam_points_per_side = gr.Slider(8, 128, value=32, step=8, label="Points Per Side")
                            sam_pred_iou_thresh = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Prediction IoU Threshold")
                            sam_stability_score_thresh = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Stability Score Threshold")

                    with gr.Row(visible=False) as row_sam_hq:
                        with gr.Column():
                            sam_hq_points_per_side = gr.Slider(8, 128, value=32, step=8, label="Points Per Side")
                            sam_hq_pred_iou_thresh = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Prediction IoU Threshold")
                            sam_hq_stability_score_thresh = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Stability Score Threshold")

                    with gr.Row(visible=False) as row_fastsam:
                        with gr.Column():
                            FASTSAM_SUBMODELS = [
                                ("FastSAM-x (YOLOv8x, best quality)", "FastSAM-x.pt"),
                                ("FastSAM-s (YOLOv8s, faster)", "FastSAM-s.pt"),
                            ]
                            fastsam_checkpoint = gr.Dropdown(
                                choices=FASTSAM_SUBMODELS,
                                value="FastSAM-x.pt",
                                label="FastSAM submodel",
                                info="Download with: python scripts/download_fastsam_checkpoints.py"
                            )
                            fastsam_conf_thresh = gr.Slider(0.1, 1.0, value=0.4, step=0.1, label="Confidence Threshold")
                            fastsam_iou_thresh = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="IoU Threshold")
                            fastsam_imgsz = gr.Slider(512, 2048, value=1024, step=256, label="Image Size")

                    with gr.Row(visible=False) as row_mobilesam:
                        with gr.Column():
                            mobilesam_points_per_side = gr.Slider(8, 128, value=32, step=8, label="Points Per Side")
                            mobilesam_pred_iou_thresh = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Prediction IoU Threshold")
                            mobilesam_stability_score_thresh = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Stability Score Threshold")

                with gr.Accordion("Mesh processing & Rendering", open=False):
                    use_modes = gr.CheckboxGroup(choices=["norms", "sdf", "matte"], value=["norms", "sdf"], label="Rendering Modes")
                    min_area = gr.Slider(1, 2000, value=256, step=1, label="Minimum Segment Area")
                    connections_bin_resolution = gr.Slider(50, 500, value=200, step=10, label="Connection Resolution")
                    connections_bin_threshold_percentage = gr.Slider(0.001, 0.2, value=0.05, step=0.001, label="Connection Threshold %")
                    smoothing_threshold_percentage_size = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001, label="Size Smoothing Threshold %")
                    smoothing_threshold_percentage_area = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001, label="Area Smoothing Threshold %")
                    smoothing_iterations = gr.Slider(1, 100, value=32, step=1, label="Smoothing Iterations")
                    repartition_lambda = gr.Slider(1, 10, value=3, step=1, label="Repartition Lambda")
                    repartition_iterations = gr.Slider(1, 10, value=2, step=1, label="Repartition Iterations")
                    target_dim = gr.Slider(256, 2048, value=1024, step=128, label="Render Resolution")
                    sampling_radius = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Camera Sampling Radius")

                segment_btn = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📊 Result")
                result_preview = gr.Model3D(
                    label="Segmented Mesh",
                    height=280
                )
                result_info = gr.Textbox(
                    label="Console",
                    lines=6,
                    max_lines=12,
                    interactive=False
                )
                download_result = gr.DownloadButton(
                    label="📥 Download Segmented Mesh",
                    value=None
                )
        
        # Event handlers
        mesh_file.change(
            fn=load_mesh_file,
            inputs=[mesh_file],
            outputs=[mesh_info, mesh_preview]
        )

        def on_model_change(choice):
            desc = MODEL_DESCRIPTIONS.get(choice, MODEL_DESCRIPTIONS["sam3"])
            return (
                desc,
                gr.update(visible=(choice == "sam2")),
                gr.update(visible=(choice == "sam3")),
                gr.update(visible=(choice == "sam")),
                gr.update(visible=(choice == "sam_hq")),
                gr.update(visible=(choice == "fastsam")),
                gr.update(visible=(choice == "mobilesam")),
            )

        model_type.change(
            fn=on_model_change,
            inputs=[model_type],
            outputs=[model_description, row_sam2, row_sam3, row_sam, row_sam_hq, row_fastsam, row_mobilesam]
        )

        # Apply preset configurations
        def apply_preset_values(preset_name):
            presets = get_preset_configs()
            if preset_name in presets:
                preset = presets[preset_name]
                return (
                    preset.get("sam3_threshold", 0.3),
                    preset.get("sam3_mask_threshold", 0.3),
                    preset.get("min_area", 256),
                    preset.get("repartition_lambda", 3),
                    preset.get("smoothing_iterations", 32)
                )
            return 0.3, 0.3, 256, 3, 32
        
        preset_dropdown.change(
            fn=apply_preset_values,
            inputs=[preset_dropdown],
            outputs=[sam3_threshold, sam3_mask_threshold, min_area, repartition_lambda, smoothing_iterations]
        )
        
        def segment_mesh_with_params_wrapper(*args):
            # Some Gradio versions send fewer values; insert FastSAM defaults if needed
            if len(args) == 31:
                args = list(args)
                # Insert after sam_hq (pos 17): fastsam_checkpoint, conf=0.4, iou=0.9, imgsz=1024
                args.insert(17, "FastSAM-x.pt")
                args.insert(18, 0.4)
                args.insert(19, 0.9)
                args.insert(20, 1024)
                args = tuple(args)
            return segment_mesh_with_params(*args)

        segment_btn.click(
            fn=segment_mesh_with_params_wrapper,
            inputs=[
                model_type,
                sam2_points_per_side, sam2_pred_iou_thresh, sam2_stability_score_thresh,
                use_smart_prompts, use_grid_prior, gemini_api_key, sam3_text_prompt, sam3_threshold, sam3_mask_threshold, sam3_points_per_side,
                sam_points_per_side, sam_pred_iou_thresh, sam_stability_score_thresh,
                sam_hq_points_per_side, sam_hq_pred_iou_thresh, sam_hq_stability_score_thresh,
                fastsam_checkpoint, fastsam_conf_thresh, fastsam_iou_thresh, fastsam_imgsz,
                mobilesam_points_per_side, mobilesam_pred_iou_thresh, mobilesam_stability_score_thresh,
                use_modes, min_area, connections_bin_resolution, connections_bin_threshold_percentage,
                smoothing_threshold_percentage_size, smoothing_threshold_percentage_area,
                smoothing_iterations, repartition_lambda, repartition_iterations,
                target_dim, sampling_radius
            ],
            outputs=[result_info, result_preview, download_result]
        )
    
    return demo

def find_free_port(start_port=7860, max_port=7870):
    """Find a free port starting from start_port"""
    import socket
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    # Create output directories
    os.makedirs("/tmp/samesh_cache", exist_ok=True)
    os.makedirs("/tmp/samesh_output", exist_ok=True)
    
    # Find available port
    port = find_free_port()
    if port is None:
        print("❌ No available ports found in range 7860-7870")
        print("💡 Kill existing Gradio apps with: pkill -f gradio")
        sys.exit(1)
    
    print(f"🎯 Starting Interactive Mesh Segmentation App on port {port}")
    print(f"🌐 Access at: http://localhost:{port}")
    print("=" * 60)
    
    # Launch the app
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=True,
        theme=gr.themes.Soft()
    )