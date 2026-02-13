#!/bin/bash
# Quick Setup Script for MLOps Cats vs Dogs Pipeline (Linux/Mac)

echo "============================================"
echo "MLOps Pipeline - Quick Setup"
echo "============================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/5] Python detected"
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created: venv/"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment and install packages
echo "[3/5] Installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Install PyTorch
echo "[4/5] Installing PyTorch..."
echo "Choose PyTorch version:"
echo "  1) CPU only (smaller download)"
echo "  2) CUDA 11.8 (NVIDIA GPU)"
echo "  3) CUDA 12.1 (NVIDIA GPU, newer)"
echo "  4) Skip (already installed)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Installing PyTorch CPU..."
        pip install torch torchvision torchaudio
        ;;
    2)
        echo "Installing PyTorch CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Installing PyTorch CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        echo "Skipping PyTorch installation"
        ;;
esac
echo ""

# Initialize Git and DVC (optional)
echo "[5/5] Initialize Git and DVC? (optional)"
echo "This will set up version control for code and data."
read -p "Initialize? (y/n): " init_git

if [[ "$init_git" =~ ^[Yy]$ ]]; then
    echo "Initializing Git..."
    git init
    git add .
    git commit -m "Initial commit: MLOps pipeline setup"
    
    echo "Initializing DVC..."
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC"
    
    echo ""
    echo "[INFO] To track the dataset with DVC, run:"
    echo "  dvc add '../archive (2)/PetImages'"
    echo "  git add PetImages.dvc .gitignore"
    echo "  git commit -m 'Track dataset with DVC'"
else
    echo "Skipping Git/DVC initialization"
fi
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To start working:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start Jupyter:"
echo "     jupyter notebook mlops_cats_dogs_pipeline.ipynb"
echo "     OR"
echo "     jupyter lab"
echo ""
echo "  3. Open the notebook and run cells sequentially"
echo ""
echo "For MLflow UI after training:"
echo "  mlflow ui --backend-store-uri file:///path/to/experiments"
echo "  Then open: http://localhost:5000"
echo ""
echo "============================================"
