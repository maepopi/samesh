#!/usr/bin/env python3
"""
Test script to verify SAM3 integration with mesh segmentation
"""
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from samesh.models.sam3 import Sam3ModelMesh
from samesh.models.sam_mesh import SamModelMesh

def test_sam3_model():
    """Test SAM3 model directly"""
    print("Testing SAM3 model directly...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test configuration
    config = OmegaConf.create({
        'sam': {
            'checkpoint': '/home/maelys/WSL_AI_HUB/TOOLS/samesh/checkpoints/sam3.pt',
            'auto': True,
            'text_prompt': 'object',
            'threshold': 0.5,
            'mask_threshold': 0.5,
            'engine_config': {
                'points_per_side': 16  # Smaller for testing
            }
        }
    })
    
    try:
        # Create SAM3 model
        sam3 = Sam3ModelMesh(config, device=device)
        print("✓ SAM3 model created successfully")
        
        # Create a simple test image
        test_image = Image.new('RGB', (256, 256), color='white')
        # Add some simple shapes
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.ellipse([50, 50, 150, 150], fill='red')
        draw.rectangle([100, 100, 200, 200], fill='blue')
        
        print("Testing with synthetic image...")
        masks = sam3(test_image)
        print(f"✓ Generated {len(masks)} masks")
        print(f"  Mask shapes: {[mask.shape for mask in masks[:3]]}")
        
        return True
        
    except Exception as e:
        print(f"✗ SAM3 model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam3_mesh_integration():
    """Test SAM3 integration with mesh segmentation pipeline"""
    print("\nTesting SAM3 integration with mesh segmentation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the updated configuration
    config_path = Path('/home/maelys/WSL_AI_HUB/TOOLS/samesh/configs/mesh_segmentation.yaml')
    
    try:
        config = OmegaConf.load(config_path)
        print(f"✓ Loaded configuration from {config_path}")
        
        # Create SamModelMesh with SAM3
        sam_mesh = SamModelMesh(config, device=device, use_sam=True)
        print("✓ SamModelMesh created with SAM3")
        
        # Check if SAM3 model was loaded
        if hasattr(sam_mesh, 'sam') and isinstance(sam_mesh.sam, Sam3ModelMesh):
            print("✓ SAM3 model correctly integrated into mesh segmentation pipeline")
            return True
        else:
            print("✗ SAM3 model not properly integrated")
            return False
            
    except Exception as e:
        print(f"✗ Mesh integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("SAM3 Integration Test Suite")
    print("=" * 50)
    
    # Check if SAM3 model exists
    sam3_path = Path('/home/maelys/WSL_AI_HUB/TOOLS/samesh/checkpoints/sam3.pt')
    if not sam3_path.exists():
        print(f"✗ SAM3 model not found at {sam3_path}")
        print("Please ensure the model was downloaded correctly.")
        return False
    
    print(f"✓ SAM3 model found at {sam3_path}")
    print(f"  Model size: {sam3_path.stat().st_size / (1024**3):.1f} GB")
    
    # Run tests
    test1_passed = test_sam3_model()
    test2_passed = test_sam3_mesh_integration()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("✓ All tests passed! SAM3 is ready for mesh segmentation.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)