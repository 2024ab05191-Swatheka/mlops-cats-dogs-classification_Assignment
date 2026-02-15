"""
Pytest configuration and fixtures
"""
import pytest
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session", autouse=True)
def create_dummy_model():
    """Create a dummy model file for testing if it doesn't exist"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'best_model.pt')
    
    # Only create if doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        
        # Create minimal dummy checkpoint
        dummy_checkpoint = {
            'model_state_dict': {},
            'epoch': 1,
            'optimizer_state_dict': {}
        }
        
        torch.save(dummy_checkpoint, model_path)
        print(f"Created dummy model at {model_path} for testing")
    
    yield
    
    # Cleanup is optional - leave the dummy model for other tests
