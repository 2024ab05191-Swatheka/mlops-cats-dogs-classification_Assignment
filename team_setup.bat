@echo off
REM Quick Setup Script for Team Members
REM Run this after cloning the repository

echo ============================================
echo MLOps Pipeline - Team Member Setup
echo ============================================
echo.

REM Check if local_config.py exists
if exist "local_config.py" (
    echo [INFO] local_config.py already exists
) else (
    echo [1/4] Creating local_config.py from template...
    copy local_config.example.py local_config.py
    echo.
    echo IMPORTANT: Edit local_config.py and update RAW_DATA_PATH
    echo with your local dataset location!
    echo.
    pause
)

REM Create virtual environment
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate and install packages
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM DVC setup
echo [4/4] DVC setup...
set /p use_dvc="Do you want to pull dataset from DVC remote? (y/n): "
if /i "%use_dvc%"=="y" (
    echo Pulling dataset from DVC remote...
    dvc pull
    echo.
    echo Dataset downloaded via DVC!
) else (
    echo.
    echo Manual dataset setup:
    echo 1. Download from: https://www.kaggle.com/c/dogs-vs-cats/data
    echo 2. Extract to your preferred location
    echo 3. Update RAW_DATA_PATH in local_config.py
)
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit local_config.py with your dataset path
echo   2. Activate environment: venv\Scripts\activate
echo   3. Start Jupyter: jupyter notebook mlops_cats_dogs_pipeline.ipynb
echo.
echo See TEAM_SETUP.md for detailed instructions
echo ============================================
pause
