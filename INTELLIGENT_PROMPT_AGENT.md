# 🧠 Intelligent Prompt Agent for SAM3 Mesh Segmentation

## Overview

The **Intelligent Prompt Agent** is an AI-powered system that automatically analyzes rendered mesh images and generates optimal text prompts for SAM3 segmentation. Instead of manually guessing prompts, the agent uses computer vision and heuristic analysis to understand the mesh geometry and create targeted prompts for maximum segmentation accuracy.

## 🎯 **How It Works**

### 1. **Image Analysis**
The agent analyzes the rendered mesh image using multiple computer vision techniques:

- **Edge Detection:** Identifies geometric complexity and sharp features
- **Contour Analysis:** Counts and analyzes distinct shapes
- **Symmetry Detection:** Measures horizontal/vertical symmetry
- **Texture Analysis:** Evaluates surface complexity
- **Aspect Ratio:** Determines object proportions
- **Fill Ratio:** Measures how much of the image is occupied

### 2. **Mesh Type Inference**
Based on visual features and filename analysis, the agent classifies the mesh into categories:

- **Furniture:** chairs, tables, desks, shelves
- **Mechanical:** gears, engines, tools, components
- **Architectural:** buildings, columns, beams
- **Organic:** trees, plants, animals, natural forms
- **Vehicle:** cars, planes, boats, wheels
- **Decorative:** vases, sculptures, ornaments
- **Geometric:** basic shapes, abstract forms

### 3. **Smart Prompt Generation**
The agent generates targeted prompts using multiple strategies:

- **Type-Specific Prompts:** Based on inferred mesh type
- **Feature-Based Prompts:** Derived from visual analysis
- **Segmentation Strategy Prompts:** Functional, structural, geometric approaches
- **Universal Fallbacks:** Generic prompts that usually work

### 4. **Prompt Ranking & Selection**
Prompts are ranked by expected effectiveness:

- **Relevance Score:** How well the prompt matches the mesh type
- **Feature Alignment:** How well it matches visual features
- **Success Probability:** Based on SAM3's typical performance
- **Specificity Balance:** Not too generic, not too specific

### 5. **Iterative Testing**
The agent tests multiple prompts automatically and selects the one that produces the most masks:

- Tests top 5 generated prompts
- Measures mask count for each prompt
- Selects the prompt with best results
- Falls back to manual prompt if needed

## 🚀 **Usage in Gradio App**

### **Enable Smart Prompts:**
1. Open the Gradio app at `http://localhost:7860`
2. Go to **"🎯 SAM3 Parameters"** tab
3. **Check** "🧠 Use Intelligent Prompt Agent"
4. Upload your mesh and run segmentation

### **Manual Override:**
- **Uncheck** the Smart Prompts option
- Enter your own text prompt in the manual field
- The agent will be bypassed

## 📊 **Example Results**

### **Furniture Detection:**
```
Input: chair.glb
Analysis: furniture type, moderate symmetry, rectangular aspect
Generated Prompts: ['chair part', 'seat component', 'furniture element', 'support structure', 'object']
Selected: 'chair part' (found 8 masks)
```

### **Mechanical Parts:**
```
Input: gear_assembly.obj  
Analysis: mechanical type, high symmetry, many contours
Generated Prompts: ['gear part', 'mechanical component', 'symmetric part', 'machine element', 'part']
Selected: 'gear part' (found 12 masks)
```

### **Architectural Elements:**
```
Input: building_facade.stl
Analysis: architectural type, high edge density, elongated
Generated Prompts: ['structural element', 'building component', 'architectural feature', 'surface', 'part']
Selected: 'structural element' (found 6 masks)
```

## 🔧 **Configuration Options**

### **In SAM3 Config:**
```yaml
sam3:
  use_smart_prompts: true    # Enable/disable intelligent agent
  text_prompt: null          # Manual override (if smart prompts disabled)
  threshold: 0.3             # Segmentation threshold
  mask_threshold: 0.3        # Mask threshold
```

### **Agent Parameters:**
- **max_prompts:** Number of prompts to generate (default: 5)
- **test_all_prompts:** Whether to test all prompts or stop at first success
- **fallback_enabled:** Whether to use fallback prompts if none work

## 🧪 **Technical Details**

### **Visual Features Extracted:**
- `edge_density`: Ratio of edge pixels to total pixels
- `num_contours`: Number of distinct shapes detected
- `aspect_ratio`: Width/height ratio
- `horizontal_symmetry`: Correlation between left/right halves
- `texture_complexity`: Standard deviation of pixel intensities
- `fill_ratio`: Proportion of non-background pixels
- `mean_brightness`: Average pixel intensity
- `brightness_std`: Brightness variation

### **Prompt Categories:**
- **Functional:** handle, grip, support, base, surface
- **Structural:** frame, beam, joint, connection, edge
- **Geometric:** flat surface, curved surface, vertex, face
- **Semantic:** part, component, section, region, area
- **Spatial:** front, back, side, top, bottom

### **Scoring Algorithm:**
```python
score = base_score + type_match_bonus + feature_alignment + generic_bonus - specificity_penalty
```

## 📈 **Performance Benefits**

### **Before (Manual Prompts):**
- ❌ Trial and error with text prompts
- ❌ Inconsistent results across different meshes
- ❌ Time-consuming prompt optimization
- ❌ User needs domain knowledge

### **After (Intelligent Agent):**
- ✅ Automatic prompt optimization
- ✅ Consistent high-quality results
- ✅ Zero manual prompt engineering
- ✅ Works for any mesh type
- ✅ Learns from visual features
- ✅ Iterative improvement

## 🎨 **Integration Examples**

### **Python API:**
```python
from samesh.models.prompt_agent import MeshPromptAgent
from PIL import Image

agent = MeshPromptAgent()
image = Image.open("rendered_mesh.png")
prompts = agent.generate_optimal_prompts(image, "chair.glb")
print(f"Best prompts: {prompts}")
```

### **Direct Usage:**
```python
from samesh.models.prompt_agent import generate_smart_prompts

prompts = generate_smart_prompts(mesh_image, "furniture.obj", max_prompts=3)
```

## 🔮 **Future Enhancements**

### **Planned Features:**
- **Deep Learning Integration:** Use pre-trained vision models for better analysis
- **Prompt Learning:** Learn from successful prompt-result pairs
- **Multi-Modal Analysis:** Combine 2D renders with 3D geometry analysis
- **User Feedback Loop:** Learn from user corrections and preferences
- **Semantic Understanding:** Use language models for better prompt generation
- **Domain Adaptation:** Specialized agents for specific industries

### **Advanced Capabilities:**
- **Context Awareness:** Consider mesh purpose and application
- **Progressive Refinement:** Iteratively improve prompts based on results
- **Ensemble Methods:** Combine multiple prompt strategies
- **Quality Metrics:** Evaluate segmentation quality automatically

## 🏆 **Benefits Summary**

1. **🎯 Accuracy:** Generates prompts specifically tailored to each mesh
2. **⚡ Speed:** Eliminates manual prompt trial-and-error
3. **🧠 Intelligence:** Uses computer vision to understand mesh geometry
4. **🔄 Adaptability:** Works across all mesh types and domains
5. **📊 Consistency:** Provides reliable results every time
6. **🎨 Creativity:** Generates diverse prompt strategies automatically

---

**The Intelligent Prompt Agent transforms SAM3 mesh segmentation from a manual art into an automated science!** 🚀