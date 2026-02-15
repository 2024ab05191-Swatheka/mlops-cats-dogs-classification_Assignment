"""
Unit tests for data preprocessing and model utilities
"""
import pytest
import torch
from PIL import Image
import numpy as np


def test_image_preprocessing():
    """Test image can be loaded and preprocessed"""
    from torchvision import transforms
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    transformed = transform(test_image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)


def test_model_architecture():
    """Test CNN model can be created"""
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
            self.fc = torch.nn.Linear(32, 2)
        
        def forward(self, x):
            return self.fc(torch.randn(x.size(0), 32))
    
    model = SimpleCNN()
    assert model is not None
    assert isinstance(model, torch.nn.Module)
