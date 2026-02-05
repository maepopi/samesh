# SAM3 Integration Guide for Samesh

This guide documents the successful integration of Meta's SAM3 (Segment Anything Model 3) into the samesh mesh segmentation pipeline.

## Overview

SAM3 is Meta's latest foundation model for promptable segmentation that introduces text-based prompting capabilities alongside the visual prompts from SAM2. This integration allows for more sophisticated mesh segmentation using natural language descriptions.

## What Was Accomplished

### 1. Model Download ✅
- Successfully downloaded SAM3 model (3.2 GB) from Hugging Face
- Model stored at: `/home/maelys/WSL_AI_HUB/TOOLS/samesh/checkpoints/sam3.pt`

### 2. Dependencies Installation ✅
- Installed official SAM3 library: `pip install git+https://github.com/facebookresearch/sam3.git`
- Installed required dependencies:
  - `einops` - for tensor operations
  - `decord` - for video processing
  - `pycocotools` - for COCO dataset utilities
  - `accelerate` - for model acceleration
  - `transformers` - for model loading
  - `timm` - for vision transformers

### 3. Code Integration ✅
- Created `src/samesh/models/sam3.py` - SAM3 model wrapper
- Updated `src/samesh/models/sam_mesh.py` - integrated SAM3 into mesh pipeline
- Updated `configs/mesh_segmentation.yaml` - added SAM3 configuration options

### 4. Configuration ✅
Updated mesh segmentation config with SAM3 options:
```yaml
sam:
  model_type: sam3  # Select SAM3 instead of SAM2
  sam3:
    checkpoint: /home/maelys/WSL_AI_HUB/TOOLS/samesh/checkpoints/sam3.pt
    auto: True
    text_prompt: "object part"  # Text-based segmentation
    threshold: 0.5
    mask_threshold: 0.5
```

### 5. Testing ✅
- Created comprehensive test suite (`test_sam3_integration.py`)
- All tests pass successfully:
  - ✅ SAM3 model loads correctly
  - ✅ Generates segmentation masks
  - ✅ Integrates with mesh segmentation pipeline

## Key Features

### Text-Based Prompting
SAM3 introduces the ability to segment objects using natural language descriptions:
- `"object part"` - segments individual parts of objects
- `"handle"` - segments handles specifically
- `"surface"` - segments surface regions
- And many more possibilities

### Backward Compatibility
The integration maintains compatibility with existing SAM2 workflows:
- Visual prompts (points, boxes) still supported
- Same API interface for mesh segmentation
- Easy switching between SAM2 and SAM3

### Enhanced Segmentation
SAM3 provides improved segmentation quality:
- Better handling of complex geometries
- More accurate part separation
- Enhanced text understanding for semantic segmentation

## Usage Examples

### Basic Usage
```python
from omegaconf import OmegaConf
from samesh.models.sam_mesh import segment_mesh

# Load configuration
config = OmegaConf.load('configs/mesh_segmentation.yaml')

# Ensure SAM3 is selected
config.sam.model_type = 'sam3'

# Run segmentation
segmented_mesh = segment_mesh(
    filename="your_mesh.glb",
    config=config,
    visualize=True
)
```

### Custom Text Prompts
```python
# Modify text prompt for specific segmentation
config.sam.sam.text_prompt = "mechanical parts"

# Or segment specific features
config.sam.sam.text_prompt = "decorative elements"
```

### Switching Between Models
```python
# Use SAM3
config.sam.model_type = 'sam3'

# Use SAM2 (legacy)
config.sam.model_type = 'sam2'
```

## File Structure

```
samesh/
├── checkpoints/
│   ├── sam2_hiera_large.pt     # SAM2 model
│   └── sam3.pt                 # SAM3 model (new)
├── configs/
│   └── mesh_segmentation.yaml  # Updated with SAM3 config
├── src/samesh/models/
│   ├── sam.py                  # Original SAM models
│   ├── sam3.py                 # SAM3 wrapper (new)
│   └── sam_mesh.py             # Updated mesh pipeline
├── test_sam3_integration.py    # Test suite (new)
├── example_sam3_mesh_segmentation.py  # Example script (new)
└── SAM3_INTEGRATION_GUIDE.md   # This guide (new)
```

## Performance Notes

- **Model Size**: SAM3 is 3.2 GB (vs 857 MB for SAM2)
- **Memory Usage**: Requires more GPU memory due to larger model
- **Speed**: Slightly slower than SAM2 but provides better quality
- **Text Processing**: Additional overhead for text prompt processing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use smaller input images
   - Consider using CPU if GPU memory is limited

2. **Missing Dependencies**
   - Ensure all dependencies are installed in samesh conda environment
   - Run: `conda activate samesh && pip install -r requirements.txt`

3. **Model Download Issues**
   - Check internet connection
   - Verify Hugging Face access (some models require authentication)
   - Manually download from: https://huggingface.co/facebook/sam3

### Environment Setup
```bash
# Activate samesh environment
conda activate samesh

# Install SAM3 and dependencies
pip install git+https://github.com/facebookresearch/sam3.git
pip install einops decord pycocotools accelerate

# Test installation
python test_sam3_integration.py
```

## Future Enhancements

1. **Video Segmentation**: Integrate SAM3's video capabilities
2. **Interactive Prompting**: Add support for interactive text refinement
3. **Batch Processing**: Optimize for processing multiple meshes
4. **Custom Prompts**: Create domain-specific prompt templates
5. **Performance Optimization**: Model quantization and optimization

## Conclusion

SAM3 has been successfully integrated into the samesh pipeline, providing enhanced mesh segmentation capabilities through text-based prompting. The integration maintains backward compatibility while offering new possibilities for semantic mesh understanding.

The system is now ready for production use with both SAM2 and SAM3 models, allowing users to choose the best model for their specific use case.