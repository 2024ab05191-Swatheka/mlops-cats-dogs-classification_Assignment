"""
Pytest configuration and fixtures
"""
import pytest
import torch
import torch.nn as nn
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Mock model class for testing
class MockBaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MockBaselineCNN, self).__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(torch.randn(x.size(0), 10))
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def load_state_dict(self, state_dict, strict=True):
        pass


@pytest.fixture(scope="session", autouse=True)
def mock_model_loading(monkeypatch):
    """Mock torch.load to avoid loading actual model"""
    def mock_load(*args, **kwargs):
        return {'model_state_dict': {}}
    
    # Patch at session level
    torch.load = mock_load
    
    yield
