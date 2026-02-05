#!/usr/bin/env python3
"""
Simple test of Gradio mesh segmentation
"""

import gradio as gr
import trimesh
import numpy as np
from pathlib import Path
import sys
import os

# Add the samesh package to the path
sys.path.append('/home/maelys/WSL_AI_HUB/TOOLS/samesh/src')

def test_mesh_load(file_path):
    """Test mesh loading"""
    if file_path is None:
        return "❌ No file selected"
    
    try:
        # Load the mesh
        mesh = trimesh.load(file_path)
        
        if isinstance(mesh, trimesh.Trimesh):
            current_mesh = mesh
        elif isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return "❌ No geometry found in the scene."
            combined = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            current_mesh = combined
        else:
            return "❌ Loaded object is neither a Trimesh nor a Scene."
        
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

def create_simple_interface():
    with gr.Blocks(title="🎯 Simple Mesh Test") as demo:
        
        gr.Markdown("# 🎯 Simple Mesh Loading Test")
        
        with gr.Row():
            with gr.Column():
                mesh_file = gr.File(
                    label="Upload Mesh File",
                    file_types=[".glb", ".obj", ".stl", ".ply"],
                    type="filepath"
                )
                
                mesh_info = gr.Textbox(
                    label="Mesh Information",
                    lines=8,
                    interactive=False
                )
            
            with gr.Column():
                mesh_preview = gr.Model3D(
                    label="Mesh Preview",
                    height=400
                )
        
        # Event handlers
        mesh_file.change(
            fn=test_mesh_load,
            inputs=[mesh_file],
            outputs=[mesh_info, mesh_preview]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_simple_interface()
    print("🚀 Starting simple test app at http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )