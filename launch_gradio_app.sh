#!/bin/bash

# 🚀 Launch Gradio Mesh Segmentation App
echo "🎯 Starting Interactive Mesh Segmentation App..."
echo "================================================"

# Activate the samesh conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate samesh

# Install gradio if not already installed
echo "📦 Checking Gradio installation..."
pip install gradio --quiet

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Stop any existing Gradio apps
echo "🧹 Cleaning up existing apps..."
pkill -f gradio_mesh_segmentation.py 2>/dev/null || true
pkill -f test_gradio_simple.py 2>/dev/null || true
sleep 2

# Launch the app
echo "🚀 Launching Gradio app (will auto-find available port)"
echo "📁 Upload a mesh file and experiment with parameters!"
echo "🔄 Press Ctrl+C to stop the app"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
python gradio_mesh_segmentation.py