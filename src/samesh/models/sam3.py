"""
SAM3 Model integration for mesh segmentation
"""
import re
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf

# Import official SAM3 library
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from samesh.data.common import NumpyTensor
from samesh.models.sam import point_grid_from_mask
from samesh.models.prompt_agent import GeminiMeshPromptAgent


class Sam3Engine:
    """
    Compatibility wrapper to make SAM3 work with existing mesh segmentation code
    """
    def __init__(self, sam3_model):
        self.sam3_model = sam3_model
        self.point_grids = None  # For compatibility with existing code

class Sam3ModelMesh(nn.Module):
    """
    SAM3 Model wrapper for mesh segmentation using text prompts and automatic mask generation
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        Initialize SAM3 model for mesh segmentation
        
        Args:
            config: Configuration containing SAM3 model settings
            device: Device to run the model on
        """
        super().__init__()
        self.config = config
        self.device = device
        
        # Load SAM3 model and processor using official library
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Create engine wrapper for compatibility
        self.engine = Sam3Engine(self)
        
        # Initialize Gemini-powered prompt agent
        api_key = config.sam.get('gemini_api_key', None)
        self.prompt_agent = GeminiMeshPromptAgent(api_key=api_key)
        
        # Configuration for automatic mask generation
        self.auto_mode = config.sam.get('auto', True)
        self.text_prompt = config.sam.get('text_prompt', None)
        self.threshold = config.sam.get('threshold', 0.5)
        self.mask_threshold = config.sam.get('mask_threshold', 0.5)
        self.use_smart_prompts = config.sam.get('use_smart_prompts', True)
        self.use_grid_prior = config.sam.get('use_grid_prior', True)  # NEW: Enable grid-based priors
        self.points_per_side = config.sam.get('engine_config', {}).get('points_per_side', 32)

    def generate_grid_points(self, image_shape: tuple) -> np.ndarray:
        """
        Generate a grid of points for automatic mask generation (SAM2-style)
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            Array of point coordinates [n, 2] in (x, y) format
        """
        h, w = image_shape[:2]
        
        # Create grid points like SAM2AutomaticMaskGenerator
        points_per_side = self.points_per_side
        
        # Generate grid coordinates
        x_coords = np.linspace(0, w - 1, points_per_side, dtype=int)
        y_coords = np.linspace(0, h - 1, points_per_side, dtype=int)
        
        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        print(f"🔲 Generated {len(grid_points)} grid points ({points_per_side}×{points_per_side})")
        return grid_points

    def process_comprehensive_sam3_masks(self, image: Image.Image, inference_state, grid_points: np.ndarray) -> dict:
        """
        Generate comprehensive masks using SAM3's native text prompting with geometric guidance
        This uses only SAM3 capabilities - no SAM2 hybrid approach
        
        Args:
            image: PIL Image
            inference_state: SAM3 inference state
            grid_points: Array of point coordinates for spatial guidance
            
        Returns:
            Dictionary with comprehensive SAM3-generated masks
        """
        print(f"🎯 SAM3 Comprehensive Mask Generation: {len(grid_points)} spatial guidance points")
        
        try:
            all_masks = []
            
            # Strategy 1: Use diverse text prompts for comprehensive coverage
            print("   📝 Using diverse SAM3 text prompts...")
            comprehensive_prompts = [
                "object", "part", "component", "surface", "element",
                "region", "area", "section", "piece", "segment",
                "feature", "structure", "detail", "portion", "fragment",
                "body", "limb", "appendage", "joint", "connection",
                "edge", "boundary", "interior", "exterior", "face"
            ]
            
            for prompt in comprehensive_prompts:
                try:
                    # Use SAM3's text prompting
                    prompt_output = self.processor.set_text_prompt(
                        prompt=prompt,
                        state=inference_state
                    )
                    
                    if prompt_output and isinstance(prompt_output, dict):
                        masks = prompt_output.get("masks", None)
                        if masks is not None and isinstance(masks, torch.Tensor) and masks.shape[0] > 0:
                            all_masks.append(masks)
                            print(f"     '{prompt}': {masks.shape[0]} masks")
                
                except Exception as e:
                    print(f"     '{prompt}' failed: {e}")
                    continue
            
            # Strategy 2: Try geometric prompts if available
            if hasattr(self.processor, 'add_geometric_prompt'):
                print("   🔷 Using SAM3 geometric prompts...")
                try:
                    # Sample some grid points for geometric guidance
                    sample_points = grid_points[::max(1, len(grid_points)//10)]  # Sample 10 points
                    
                    for i, point in enumerate(sample_points):
                        try:
                            # Add geometric prompt at this location
                            geom_output = self.processor.add_geometric_prompt(
                                state=inference_state,
                                point=point.tolist()
                            )
                            
                            if geom_output and isinstance(geom_output, dict):
                                masks = geom_output.get("masks", None)
                                if masks is not None and isinstance(masks, torch.Tensor) and masks.shape[0] > 0:
                                    all_masks.append(masks)
                                    print(f"     Geometric point {i+1}: {masks.shape[0]} masks")
                        
                        except Exception as e:
                            print(f"     Geometric point {i+1} failed: {e}")
                            continue
                            
                except Exception as e:
                    print(f"   Geometric prompts failed: {e}")
            
            # Combine all SAM3 masks
            if all_masks:
                combined_masks = torch.cat(all_masks, dim=0)
                print(f"✅ SAM3 Comprehensive Generation: {combined_masks.shape[0]} total masks")
                
                return {
                    "masks": combined_masks,
                    "scores": None
                }
            else:
                print("❌ No masks generated from SAM3 comprehensive approach")
                return {"masks": torch.empty(0, 1, *image.size[::-1]), "scores": None}
            
        except Exception as e:
            print(f"❌ SAM3 comprehensive processing error: {e}")
            return {"masks": torch.empty(0, 1, *image.size[::-1]), "scores": None}

    def process_image(self, image: Image, prompt: dict = None) -> NumpyTensor['n h w']:
        """
        Process image with SAM3 to generate segmentation masks
        Compatible with SAM2 interface - returns masks in format [n, h, w]
        
        Args:
            image: PIL Image to segment
            prompt: Optional prompt dictionary (for compatibility with SAM2)
            
        Returns:
            Array of binary masks [n, h, w] - same format as SAM2
        """
        try:
            # Convert PIL image to numpy for processing
            image_array = np.array(image)
            h, w = image_array.shape[:2]
            
            # Set image for processing
            inference_state = self.processor.set_image(image)
            
            # SAM3 COMPREHENSIVE APPROACH: Use SAM3's native capabilities for comprehensive coverage
            comprehensive_masks = None
            if self.use_grid_prior:
                print("🔲 SAM3 Comprehensive Mask Generation...")
                grid_points = self.generate_grid_points((h, w))
                comprehensive_output = self.process_comprehensive_sam3_masks(image, inference_state, grid_points)
                comprehensive_masks = comprehensive_output.get("masks", None)
            
            # Determine text prompt using intelligent agent
            if self.use_smart_prompts and not self.text_prompt:
                print("🧠 Analyzing image with Prompt Agent...")
                smart_prompts = self.prompt_agent.generate_optimal_prompts(image, max_prompts=5)
                
                # Try each smart prompt until one works
                best_prompt = None
                best_output = None
                best_mask_count = 0
                
                for prompt in smart_prompts:
                    print(f"🎯 Trying smart prompt: '{prompt}'")
                    try:
                        test_output = self.processor.set_text_prompt(
                            state=inference_state,
                            prompt=prompt
                        )
                        
                        if test_output and isinstance(test_output, dict):
                            test_masks = test_output.get("masks", None)
                            if test_masks is not None and isinstance(test_masks, torch.Tensor):
                                mask_count = test_masks.shape[0]
                                print(f"   Found {mask_count} masks")
                                
                                if mask_count > best_mask_count:
                                    best_mask_count = mask_count
                                    best_prompt = prompt
                                    best_output = test_output
                                    
                                    # If we found a good number of masks, use this prompt
                                    if mask_count >= 3:
                                        break
                    except Exception as e:
                        print(f"   Error with prompt '{prompt}': {e}")
                        continue
                
                if best_prompt and best_mask_count > 0:
                    text_prompt = best_prompt
                    output = best_output
                    print(f"✅ Selected best prompt: '{text_prompt}' ({best_mask_count} masks)")
                else:
                    # Fallback to manual prompt or default
                    text_prompt = self.text_prompt if self.text_prompt else "object"
                    print(f"🔄 Using fallback prompt: '{text_prompt}'")
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt
                    )
            else:
                # Use manual prompt or default
                text_prompt = self.text_prompt if self.text_prompt else "object"
                print(f"📝 Using manual prompt: '{text_prompt}'")
                
                # Run SAM3 segmentation with text prompt
                output = self.processor.set_text_prompt(
                    state=inference_state, 
                    prompt=text_prompt
                )
            
        except Exception as e:
            print(f"SAM3 processing failed: {e}")
            # Return single dummy mask in correct format
            h, w = np.array(image).shape[:2]
            return np.zeros((1, h, w), dtype=bool)
        
        # Process SAM3 output to match SAM2 format
        if output is None or not isinstance(output, dict):
            print("SAM3 returned no output, creating dummy mask")
            h, w = np.array(image).shape[:2]
            return np.zeros((1, h, w), dtype=bool)
        
        # Extract masks from SAM3 text output
        text_masks = output.get("masks", None)
        
        # COMBINE ALL SAM3 MASKS
        all_raw_masks = []
        
        # Add SAM3 comprehensive masks (diverse text + geometric prompts)
        if comprehensive_masks is not None and isinstance(comprehensive_masks, torch.Tensor) and comprehensive_masks.shape[0] > 0:
            all_raw_masks.append(comprehensive_masks)
            print(f"🔲 Added {comprehensive_masks.shape[0]} SAM3 comprehensive masks")
        
        # Add SAM3 targeted text-based masks (smart prompt-based)
        if text_masks is not None and isinstance(text_masks, torch.Tensor) and text_masks.shape[0] > 0:
            all_raw_masks.append(text_masks)
            print(f"📝 Added {text_masks.shape[0]} SAM3 targeted text masks")
        
        # Combine all SAM3 masks
        if all_raw_masks:
            raw_masks = torch.cat(all_raw_masks, dim=0)
            print(f"🎯 SAM3 Total: {raw_masks.shape[0]} masks (comprehensive + targeted)")
        else:
            raw_masks = None
        
        # Handle case with no masks - create fallback masks
        if raw_masks is None or (isinstance(raw_masks, torch.Tensor) and raw_masks.shape[0] == 0):
            print(f"SAM3 found no masks with prompt '{text_prompt}', creating fallback masks")
            h, w = np.array(image).shape[:2]
            
            # Create multiple fallback masks based on image regions
            fallback_masks = []
            
            # Mask 1: Center region
            center_mask = np.zeros((h, w), dtype=bool)
            center_h, center_w = h // 4, w // 4
            center_mask[center_h:h-center_h, center_w:w-center_w] = True
            fallback_masks.append(center_mask)
            
            # Mask 2: Left half
            left_mask = np.zeros((h, w), dtype=bool)
            left_mask[:, :w//2] = True
            fallback_masks.append(left_mask)
            
            # Mask 3: Right half  
            right_mask = np.zeros((h, w), dtype=bool)
            right_mask[:, w//2:] = True
            fallback_masks.append(right_mask)
            
            # Mask 4: Top half
            top_mask = np.zeros((h, w), dtype=bool)
            top_mask[:h//2, :] = True
            fallback_masks.append(top_mask)
            
            # Mask 5: Bottom half
            bottom_mask = np.zeros((h, w), dtype=bool)
            bottom_mask[h//2:, :] = True
            fallback_masks.append(bottom_mask)
            
            print(f"Created {len(fallback_masks)} fallback masks")
            return np.stack(fallback_masks, axis=0)
        
        # Convert tensor to list for processing
        if isinstance(raw_masks, torch.Tensor):
            raw_masks = [raw_masks[i] for i in range(raw_masks.shape[0])]
        elif not isinstance(raw_masks, (list, tuple)):
            raw_masks = [raw_masks]
        
        # Convert masks to numpy and ensure consistent format
        processed_masks = []
        
        for i, mask in enumerate(raw_masks):
            try:
                # Convert tensor to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Handle different dimensionalities
                if mask.ndim == 4:  # [1, 1, h, w] or [b, c, h, w]
                    mask = mask[0, 0]  # Take first batch and channel
                elif mask.ndim == 3:
                    if mask.shape[0] == 1:  # [1, h, w]
                        mask = mask[0]
                    elif mask.shape[2] == 1:  # [h, w, 1]
                        mask = mask[:, :, 0]
                    else:  # [c, h, w] - take first channel
                        mask = mask[0]
                elif mask.ndim == 2:  # [h, w] - already correct
                    pass
                else:
                    print(f"Unexpected mask dimensions: {mask.shape}")
                    continue
                
                # Ensure mask is 2D and correct size
                if mask.shape != (h, w):
                    print(f"Resizing mask from {mask.shape} to ({h}, {w})")
                    # Simple resize by cropping or padding
                    resized_mask = np.zeros((h, w), dtype=bool)
                    min_h, min_w = min(h, mask.shape[0]), min(w, mask.shape[1])
                    resized_mask[:min_h, :min_w] = mask[:min_h, :min_w]
                    mask = resized_mask
                
                # Convert to boolean and add to processed masks
                processed_masks.append(mask.astype(bool))
                
            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                continue
        
        # Ensure we have at least one mask
        if len(processed_masks) == 0:
            print("No valid masks processed, creating dummy mask")
            processed_masks = [np.zeros((h, w), dtype=bool)]
        
        # Stack masks to create [n, h, w] format (same as SAM2)
        try:
            masks = np.stack(processed_masks, axis=0)
            print(f"SAM3 processed {len(masks)} masks with shape {masks.shape}")
        except Exception as e:
            print(f"Error stacking masks: {e}")
            # Fallback: create single dummy mask
            masks = np.zeros((1, h, w), dtype=bool)
        
        # Sort by area (largest first) for consistency with SAM2
        if len(masks) > 1:
            areas = [mask.sum() for mask in masks]
            sorted_indices = np.argsort(areas)[::-1]
            masks = masks[sorted_indices]
        
        return masks

    def forward(self, image: Image, texts: list[str] = None) -> NumpyTensor['n h w']:
        """
        Forward pass for SAM3 model
        
        Args:
            image: PIL Image to segment
            texts: Optional list of text prompts (uses first one if provided)
            
        Returns:
            Array of binary masks [n, h, w]
        """
        if texts and len(texts) > 0:
            # Use text prompt
            self.text_prompt = texts[0]
        
        return self.process_image(image)


# Compatibility function to create SAM3 model with same interface as SAM2
def create_sam3_model(config: OmegaConf, device='cuda') -> Sam3ModelMesh:
    """
    Create SAM3 model with configuration
    
    Args:
        config: Configuration object
        device: Device to run model on
        
    Returns:
        SAM3 model instance
    """
    return Sam3ModelMesh(config, device)


if __name__ == '__main__':
    import time
    from pathlib import Path

    device = 'cuda'
    
    _repo_root = Path(__file__).resolve().parents[3]  # src/samesh/models -> repo root
    # Test configuration
    config = OmegaConf.create({
        'sam': {
            'checkpoint': str(_repo_root / 'checkpoints' / 'sam3.pt'),
            'auto': True,
            'text_prompt': 'object',
            'threshold': 0.5,
            'mask_threshold': 0.5,
            'engine_config': {'points_per_side': 32}
        }
    })

    test_image_path = _repo_root / 'assets' / 'samesh_examples.png'
    if test_image_path.exists():
        image = Image.open(test_image_path).convert('RGB')
        
        sam3 = Sam3ModelMesh(config, device)
        start_time = time.time()
        masks = sam3(image)
        print(f'SAM3 Elapsed time: {time.time() - start_time:.2f} s')
        print(f'Generated {len(masks)} masks')
        print(f'Mask shapes: {[mask.shape for mask in masks[:3]]}')  # Show first 3 mask shapes
    else:
        print(f"Test image not found at {test_image_path}")
        print("Please provide a valid image path for testing")