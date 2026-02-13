@echo off
REM Quick Setup Script for MLOps Cats vs Dogs Pipeline
REM Run this script to set up the environment

echo ============================================
echo MLOps Pipeline - Quick Setup
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python detected
python --version
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created: venv\
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment and install packages
echo [3/5] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM Install PyTorch
echo [4/5] Installing PyTorch...
echo Choose PyTorch version:
echo   1) CPU only (smaller download)
echo   2) CUDA 11.8 (NVIDIA GPU)
echo   3) CUDA 12.1 (NVIDIA GPU, newer)
echo   4) Skip (already installed)
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Installing PyTorch CPU...
    pip install torch torchvision torchaudio
) else if "%choice%"=="2" (
    echo Installing PyTorch CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="3" (
    echo Installing PyTorch CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Skipping PyTorch installation
)
echo.

REM Initialize Git and DVC (optional)
echo [5/5] Initialize Git and DVC? (optional)
echo This will set up version control for code and data.
set /p init_git="Initialize? (y/n): "

if /i "%init_git%"=="y" (
    echo Initializing Git...
    git init
    git add .
    git commit -m "Initial commit: MLOps pipeline setup"
    
    echo Initializing DVC...
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC"
    
    echo.
    echo [INFO] To track the dataset with DVC, run:
    echo   dvc add "..\archive (2)\PetImages"
    echo   git add PetImages.dvc .gitignore
    echo   git commit -m "Track dataset with DVC"
) else (
    echo Skipping Git/DVC initialization
)
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To start working:
echo   1. Activate virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Start Jupyter:
echo      jupyter notebook mlops_cats_dogs_pipeline.ipynb
echo      OR
echo      jupyter lab
echo.
echo   3. Open the notebook and run cells sequentially
echo.
echo For MLflow UI after training:
echo   mlflow ui --backend-store-uri file:///C:/Users/swath/dataset/mlops_project/experiments
echo   Then open: http://localhost:5000
echo.
echo ============================================
pause
