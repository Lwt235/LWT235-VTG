#!/bin/bash
# LWT235-VTG Environment Installation Script

set -e

echo "================================================="
echo "  LWT235-VTG Environment Setup"
echo "================================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="${VTG_ENV_NAME:-vtg_env}"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 -y

echo "Activating environment..."
# Source conda to make activate available
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "================================================="
echo "  Installation Complete!"
echo "================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify the installation, run:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
