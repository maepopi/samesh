#!/usr/bin/env python3
"""
Test FastSAM mesh segmentation with both FastSAM-x and FastSAM-s models.
This script isolates FastSAM to ensure it works correctly before using in the Gradio app.
"""

import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

# PyTorch 2.6+ weights_only=True: patch torch.load to use weights_only=False for Ultralytics/FastSAM
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to default to weights_only=False for Ultralytics/FastSAM checkpoints"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add samesh to path
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from samesh.models.sam_mesh import segment_mesh
from samesh.data.loaders import read_mesh
import trimesh

def test_fastsam_model(checkpoint_name: str, mesh_path: Path):
    """Test a single FastSAM model checkpoint"""
    print(f"\n{'='*80}")
    print(f"Testing FastSAM: {checkpoint_name}")
    print(f"{'='*80}")
    
    # Create minimal config for FastSAM
    config = {
        'cache': '/tmp/samesh_fastsam_test',
        'cache_overwrite': True,
        'output': '/tmp/samesh_fastsam_test_output',
        
        'sam': {
            'model_type': 'fastsam',
            'sam': {
                'checkpoint': str(_REPO_ROOT / 'checkpoints' / checkpoint_name),
                'auto': True,
                'engine_config': {
                    'points_per_side': 32,  # Required by schema but not used by FastSAM
                    'conf': 0.4,
                    'iou': 0.9,
                    'imgsz': 1024,
                }
            }
        },
        
        'sam_mesh': {
            'use_modes': ['norms', 'sdf'],
            'min_area': 256,
            'connections_bin_resolution': 200,
            'connections_bin_threshold_percentage': 0.05,
            'smoothing_threshold_percentage_size': 0.01,
            'smoothing_threshold_percentage_area': 0.01,
            'smoothing_iterations': 32,
            'repartition_lambda': 3,
            'repartition_iterations': 2,
        },
        
        'renderer': {
            'target_dim': [1024, 1024],
            'camera_generation_method': 'icosahedron',
            'renderer_args': {
                'interpolate_norms': True
            },
            'sampling_args': {'radius': 2.5},
            'lighting_args': {}
        }
    }
    
    config = OmegaConf.create(config)
    
    print(f"✓ Config created")
    print(f"  Checkpoint: {config.sam.sam.checkpoint}")
    print(f"  Confidence: {config.sam.sam.engine_config.conf}")
    print(f"  IoU: {config.sam.sam.engine_config.iou}")
    print(f"  Image size: {config.sam.sam.engine_config.imgsz}")
    
    # Check if checkpoint exists
    checkpoint_path = Path(config.sam.sam.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run: python scripts/download_fastsam_checkpoints.py")
        return False
    
    print(f"✓ Checkpoint exists ({checkpoint_path.stat().st_size / (1024*1024):.1f} MB)")
    
    # Load mesh
    try:
        print(f"\nLoading mesh: {mesh_path}")
        mesh = read_mesh(str(mesh_path))
        print(f"✓ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"❌ Failed to load mesh: {e}")
        return False
    
    # Export mesh temporarily (segment_mesh expects a path)
    temp_mesh_path = Path("/tmp/test_fastsam_mesh.obj")
    try:
        mesh.export(temp_mesh_path)
        print(f"✓ Mesh exported to {temp_mesh_path}")
    except Exception as e:
        print(f"❌ Failed to export mesh: {e}")
        return False
    
    # Run segmentation
    try:
        print(f"\nRunning FastSAM segmentation...")
        print(f"  This may take a minute...")
        segmented_mesh = segment_mesh(str(temp_mesh_path), config)
        print(f"✓ Segmentation complete!")
        
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
        
        print(f"\n✅ SUCCESS!")
        print(f"  Segments found: {num_segments}")
        print(f"  Original faces: {len(mesh.faces):,}")
        print(f"  Segmented faces: {len(segmented_mesh.faces):,}")
        
        # Save result
        output_path = _REPO_ROOT / f"test_fastsam_{checkpoint_name.replace('.pt', '')}.glb"
        segmented_mesh.export(output_path)
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("FastSAM Mesh Segmentation Test")
    print("=" * 80)
    
    # Find or create a test mesh
    test_meshes = [
        _REPO_ROOT / "canap_trellis_quadremesher.glb",
        _REPO_ROOT / "canap_trellis_quadremesher_combined.glb",
        _REPO_ROOT / "assets" / "ice_cream.glb",
        _REPO_ROOT / "assets" / "potion.glb",
        _REPO_ROOT / "assets" / "jacket.glb",
    ]
    
    mesh_path = None
    for path in test_meshes:
        if path.exists():
            mesh_path = path
            break
    
    if mesh_path is None:
        print("No existing test meshes found. Creating a test mesh...")
        # Create a simple combined shape (sphere + box) for testing
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        sphere.apply_translation([0, 0, 1.5])
        
        box = trimesh.creation.box(extents=[2.0, 2.0, 1.0])
        
        test_mesh = trimesh.util.concatenate([sphere, box])
        
        mesh_path = Path("/tmp/test_fastsam_mesh.obj")
        test_mesh.export(mesh_path)
        print(f"✓ Created test mesh: {mesh_path}")
        print(f"  Vertices: {len(test_mesh.vertices)}, Faces: {len(test_mesh.faces)}")
    else:
        print(f"Using existing test mesh: {mesh_path}")
    
    # Test both FastSAM models
    models = [
        "FastSAM-x.pt",
        "FastSAM-s.pt",
    ]
    
    results = {}
    for model in models:
        results[model] = test_fastsam_model(model, mesh_path)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*80}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
