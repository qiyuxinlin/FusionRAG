#!/bin/bash
# Installation script for ktransformers-dev

set -e  # Exit on error

echo "=========================================="
echo "Installing ktransformers-dev dependencies"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Detect Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Detected Python version: $PYTHON_VERSION"

# Install faiss-cpu using conda
echo ""
echo "Step 1/4: Installing faiss-cpu with conda..."
conda install -c conda-forge faiss-cpu -y

# Install transformers with specific version
echo ""
echo "Step 2/4: Installing transformers==4.53.3..."
pip install transformers==4.53.3

# Install rouge
echo ""
echo "Step 3/4: Installing rouge..."
pip install rouge

# Install FlagEmbedding
echo ""
echo "Step 4/4: Installing FlagEmbedding..."
pip install FlagEmbedding

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - faiss-cpu (via conda)"
echo "  - transformers 4.53.3"
echo "  - rouge"
echo "  - FlagEmbedding"
echo ""
echo "You can now run the code with:"
echo "  python ktransformers/unified_process_cache.py"
