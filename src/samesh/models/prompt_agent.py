#!/usr/bin/env python3
"""
Intelligent Prompt Generation Agent for SAM3 Mesh Segmentation

This agent uses Gemini 2.5 Flash to analyze rendered mesh images and generates 
optimal text prompts for SAM3 to achieve the best possible segmentation results.
"""

import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import base64
import io
import os
import google.genai as genai

class GeminiMeshPromptAgent:
    """
    Intelligent agent that uses Gemini 2.5 Flash to analyze mesh renderings and generate optimal text prompts for SAM3
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini-powered prompt generation agent
        
        Args:
            api_key: Google AI API key. If None, will try to get from environment variable GOOGLE_AI_API_KEY
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        
        if not self.api_key:
            print("⚠️ Warning: No Google AI API key found. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter.")
            print("   Falling back to basic prompt generation.")
            self.use_gemini = False
        else:
            try:
                # Configure Gemini with new API
                self.client = genai.Client(api_key=self.api_key)
                self.use_gemini = True
                print("✅ Gemini 2.0 Flash initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                print("   Falling back to basic prompt generation.")
                self.use_gemini = False
        
        # Fallback prompts for when Gemini is not available
        self.fallback_prompts = [
            "object part", "component", "surface", "element", "section",
            "individual part", "distinct component", "separate element"
        ]

    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for Gemini API
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_with_gemini(self, image: Image.Image, filename: str = "") -> Dict:
        """
        Use Gemini 2.5 Flash to analyze the mesh image and generate segmentation prompts
        
        Args:
            image: PIL Image of rendered mesh
            filename: Optional filename for context
            
        Returns:
            Dictionary containing analysis and prompts
        """
        if not self.use_gemini:
            return self.fallback_analysis(filename)
        
        try:
            # Create the analysis prompt for Gemini
            analysis_prompt = f"""
You are an expert in 3D mesh analysis and computer vision segmentation. Analyze this rendered 3D mesh image and provide detailed segmentation guidance for SAM (Segment Anything Model).

Image Context: {f"Filename: {filename}" if filename else "No filename provided"}

Please provide a JSON response with the following structure:
{{
    "object_type": "What type of object this is (e.g., chair, car, building, tool, etc.)",
    "description": "Detailed description of the object and its visible parts",
    "main_components": ["List of 3-5 main visible components/parts"],
    "segmentation_prompts": [
        "List of 8-12 specific text prompts that would help SAM segment different parts",
        "Focus on prompts that would separate distinct functional or structural parts",
        "Use clear, specific terms like 'chair leg', 'handle', 'surface', 'edge', etc.",
        "Avoid overly generic terms like 'object' or 'thing'"
    ],
    "priority_prompts": [
        "Top 5 most important prompts from the segmentation_prompts list",
        "These should be the most likely to produce good segmentation results"
    ],
    "segmentation_strategy": "Brief explanation of the best approach to segment this object"
}}

Focus on creating prompts that will help SAM identify and separate:
1. Functional parts (handles, legs, surfaces, etc.)
2. Structural elements (frames, joints, connections)
3. Geometric features (flat surfaces, curved areas, edges)
4. Distinct components that should be separate masks

Be specific and practical - these prompts will be used directly with SAM3 for automatic segmentation.
"""

            # Send to Gemini using new API
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[
                    {'role': 'user', 'parts': [
                        {'text': analysis_prompt},
                        {'inline_data': {'mime_type': 'image/png', 'data': self.image_to_base64(image)}}
                    ]}
                ]
            )
            
            # Parse the response
            response_text = response.candidates[0].content.parts[0].text.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    
                    print(f"🧠 Gemini Analysis Complete:")
                    print(f"   Object Type: {analysis.get('object_type', 'Unknown')}")
                    print(f"   Components: {len(analysis.get('main_components', []))}")
                    print(f"   Generated Prompts: {len(analysis.get('segmentation_prompts', []))}")
                    
                    return analysis
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Failed to parse Gemini JSON response: {e}")
                print(f"Raw response: {response_text[:200]}...")
                
                # Try to extract prompts from text response as fallback
                lines = response_text.split('\n')
                prompts = []
                for line in lines:
                    line = line.strip()
                    if any(keyword in line.lower() for keyword in ['prompt', 'segment', 'part', 'component']):
                        # Extract potential prompts from the line
                        if '"' in line:
                            parts = line.split('"')
                            for part in parts[1::2]:  # Every other part starting from index 1
                                if len(part.strip()) > 2:
                                    prompts.append(part.strip())
                
                if prompts:
                    return {
                        'object_type': 'unknown',
                        'segmentation_prompts': prompts[:10],
                        'priority_prompts': prompts[:5]
                    }
                else:
                    return self.fallback_analysis(filename)
                    
        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            return self.fallback_analysis(filename)

    def fallback_analysis(self, filename: str = "") -> Dict:
        """
        Fallback analysis when Gemini is not available
        
        Args:
            filename: Optional filename for basic analysis
            
        Returns:
            Basic analysis dictionary
        """
        # Simple filename-based analysis
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['chair', 'seat', 'furniture']):
            object_type = 'chair'
            prompts = ['chair leg', 'seat', 'backrest', 'armrest', 'support', 'furniture part', 'surface']
        elif any(word in filename_lower for word in ['car', 'vehicle', 'auto']):
            object_type = 'vehicle'
            prompts = ['wheel', 'door', 'window', 'bumper', 'body panel', 'vehicle part', 'surface']
        elif any(word in filename_lower for word in ['building', 'house', 'structure']):
            object_type = 'building'
            prompts = ['wall', 'roof', 'window', 'door', 'structural element', 'surface', 'architectural feature']
        elif any(word in filename_lower for word in ['gear', 'machine', 'tool', 'mechanical']):
            object_type = 'mechanical'
            prompts = ['gear', 'component', 'mechanical part', 'surface', 'edge', 'structural element']
        else:
            object_type = 'object'
            prompts = self.fallback_prompts
        
        return {
            'object_type': object_type,
            'segmentation_prompts': prompts,
            'priority_prompts': prompts[:5]
        }

    def generate_optimal_prompts(self, image: Image.Image, filename: str = "", max_prompts: int = 5) -> List[str]:
        """
        Main method to generate optimal prompts for a mesh image using Gemini 2.5 Flash
        
        Args:
            image: PIL Image of rendered mesh
            filename: Optional filename for context
            max_prompts: Maximum number of prompts to return
            
        Returns:
            List of optimal text prompts ordered by expected effectiveness
        """
        # Analyze image with Gemini
        analysis = self.analyze_with_gemini(image, filename)
        
        # Extract prompts from analysis
        priority_prompts = analysis.get('priority_prompts', [])
        all_prompts = analysis.get('segmentation_prompts', [])
        
        # Combine priority prompts with other prompts
        optimal_prompts = []
        
        # Add priority prompts first
        for prompt in priority_prompts:
            if len(optimal_prompts) < max_prompts:
                optimal_prompts.append(prompt)
        
        # Fill remaining slots with other prompts
        for prompt in all_prompts:
            if len(optimal_prompts) < max_prompts and prompt not in optimal_prompts:
                optimal_prompts.append(prompt)
        
        # Add fallback prompts if needed
        if len(optimal_prompts) < max_prompts:
            for prompt in self.fallback_prompts:
                if len(optimal_prompts) < max_prompts and prompt not in optimal_prompts:
                    optimal_prompts.append(prompt)
        
        # Log analysis results
        object_type = analysis.get('object_type', 'unknown')
        print(f"🧠 Gemini Prompt Analysis:")
        print(f"   Object Type: {object_type}")
        print(f"   Generated {len(optimal_prompts)} optimal prompts: {optimal_prompts}")
        
        return optimal_prompts

    def create_analysis_report(self, image: Image.Image, filename: str = "") -> Dict:
        """
        Create a detailed analysis report using Gemini for debugging and optimization
        
        Args:
            image: PIL Image of rendered mesh
            filename: Optional filename for context
            
        Returns:
            Dictionary containing detailed Gemini analysis
        """
        analysis = self.analyze_with_gemini(image, filename)
        
        return {
            'filename': filename,
            'gemini_analysis': analysis,
            'object_type': analysis.get('object_type', 'unknown'),
            'description': analysis.get('description', ''),
            'main_components': analysis.get('main_components', []),
            'all_prompts': analysis.get('segmentation_prompts', []),
            'optimal_prompts': analysis.get('priority_prompts', []),
            'strategy': analysis.get('segmentation_strategy', '')
        }


# Convenience function for easy integration
def generate_smart_prompts(image: Image.Image, filename: str = "", max_prompts: int = 5, api_key: Optional[str] = None) -> List[str]:
    """
    Convenience function to generate optimal prompts for mesh segmentation using Gemini
    
    Args:
        image: PIL Image of rendered mesh
        filename: Optional filename for context
        max_prompts: Maximum number of prompts to return
        api_key: Optional Google AI API key
        
    Returns:
        List of optimal text prompts
    """
    agent = GeminiMeshPromptAgent(api_key=api_key)
    return agent.generate_optimal_prompts(image, filename, max_prompts)

# Backward compatibility alias
MeshPromptAgent = GeminiMeshPromptAgent


if __name__ == '__main__':
    # Test the Gemini prompt agent
    import sys
    from pathlib import Path
    
    print("🧪 Testing Gemini Mesh Prompt Agent")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("⚠️ Set GOOGLE_AI_API_KEY environment variable to test Gemini functionality")
        print("   Testing with fallback mode...")
    
    # Create a test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # Add some geometric shapes for testing
    import numpy as np
    img_array = np.array(test_image)
    
    # Draw a chair-like structure
    img_array[100:200, 100:120] = [139, 69, 19]  # Chair back
    img_array[120:140, 80:180] = [139, 69, 19]   # Chair seat
    img_array[140:220, 85:95] = [101, 67, 33]    # Legs
    img_array[140:220, 165:175] = [101, 67, 33]
    
    test_image = Image.fromarray(img_array)
    
    # Test the agent
    agent = GeminiMeshPromptAgent()
    prompts = agent.generate_optimal_prompts(test_image, "test_chair.glb")
    
    print(f"\n📝 Generated prompts: {prompts}")
    
    # Test analysis report
    report = agent.create_analysis_report(test_image, "test_chair.glb")
    print(f"\n📊 Analysis Report:")
    print(f"   Object Type: {report.get('object_type', 'unknown')}")
    print(f"   Components: {report.get('main_components', [])}")
    print(f"   Strategy: {report.get('strategy', 'N/A')}")
    
    print("\n✅ Gemini Prompt Agent test complete!")