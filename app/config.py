"""
Configuration file for MLOps Cats vs Dogs Pipeline
For team collaboration: Each team member should create a local_config.py 
with their specific paths (this file is gitignored)
"""

import os
from datetime import datetime

# Default configuration
class Config:
    # Dataset paths - TEAM MEMBERS: Update these for your local setup
    # Or create a local_config.py file (see local_config.example.py)
    RAW_DATA_PATH = os.environ.get('DATASET_PATH', r'C:\Users\swath\dataset\archive (2)\PetImages')
    
    # Project paths (relative paths work for all team members)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
    EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, 'experiments')
    
    # Model hyperparameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # MLflow tracking
    EXPERIMENT_NAME = 'cats_dogs_classification'
    RUN_NAME = f'baseline_cnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Try to import local configuration (for team members)
try:
    from local_config import LocalConfig
    # Override with local settings
    for attr in dir(LocalConfig):
        if not attr.startswith('_'):
            setattr(Config, attr, getattr(LocalConfig, attr))
    print("✓ Using local_config.py")
except ImportError:
    print("ℹ Using default config.py (create local_config.py for custom paths)")
