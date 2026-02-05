#!/usr/bin/env python3
"""
Example script demonstrating SAM3 mesh segmentation
"""
import sys
from pathlib import Path

# Add the src directory to the path
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root / 'src'))

import torch
from omegaconf import OmegaConf
from samesh.models.sam_mesh import segment_mesh

def main():
    """Run SAM3 mesh segmentation example"""
    print("SAM3 Mesh Segmentation Example")
    print("=" * 50)

    # Check for available mesh files (relative to repo root)
    mesh_files = [
        _repo_root / "canap_trellis_quadremesher.glb",
        _repo_root / "canap_trellis_quadremesher_combined.glb",
        _repo_root / "assets" / "example.glb",
    ]

    mesh_file = None
    for f in mesh_files:
        if f.exists():
            mesh_file = f
            break

    if mesh_file is None:
        print("No mesh files found. Place a .glb in repo root or assets/")
        return False

    print(f"Using mesh file: {mesh_file}")

    config_path = _repo_root / "configs" / "mesh_segmentation.yaml"
    config = OmegaConf.load(config_path)
    
    # Ensure we're using SAM3
    config.sam.model_type = 'sam3'
    
    print(f"Configuration loaded:")
    print(f"  - Model type: {config.sam.model_type}")
    print(f"  - Text prompt: {config.sam.sam.text_prompt}")
    print(f"  - Cache: {config.cache}")
    print(f"  - Output: {config.output}")
    
    try:
        print("\nStarting mesh segmentation with SAM3...")
        
        # Run segmentation
        segmented_mesh = segment_mesh(
            filename=mesh_file,
            config=config,
            visualize=True,  # Create visualization
            extension='glb'
        )
        
        print(f"✓ Segmentation completed successfully!")
        print(f"  - Segmented mesh saved to: {config.output}")
        print(f"  - Visualization saved to: {config.output}/{mesh_file.stem}_visualized")
        
        # Check output files
        output_dir = Path(config.output) / mesh_file.stem
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            print(f"  - Output files ({len(output_files)}):")
            for f in sorted(output_files)[:10]:  # Show first 10 files
                print(f"    - {f.name}")
            if len(output_files) > 10:
                print(f"    ... and {len(output_files) - 10} more files")
        
        return True
        
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n" + "=" * 50)
        print("✓ SAM3 mesh segmentation example completed successfully!")
        print("You can now use SAM3 for mesh segmentation in your samesh pipeline.")
    else:
        print("\n" + "=" * 50)
        print("✗ Example failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)