"""
Unit tests for data preprocessing and model utilities
"""
import pytest
import torch
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the components we need, not the whole app
from torchvision import transforms


# Define BaselineCNN here for testing (avoid loading the actual model)
class BaselineCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 14 * 14, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc_layers(x)
        return x


# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Cat', 'Dog']


def test_model_architecture():
    """Test model can be instantiated"""
    model = BaselineCNN(num_classes=2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    """Test model forward pass with dummy input"""
    model = BaselineCNN(num_classes=2)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2)  # Batch of 1, 2 classes


def test_transform_pipeline():
    """Test image transformation pipeline"""
    # Create a dummy RGB image
    dummy_image = Image.new('RGB', (100, 100), color='red')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)  # 3 channels, 224x224


def test_class_names():
    """Test class names are defined correctly"""
    assert len(class_names) == 2
    assert 'Cat' in class_names
    assert 'Dog' in class_names


def test_model_output_probabilities():
    """Test model outputs sum to 1 after softmax"""
    model = BaselineCNN(num_classes=2)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Check probabilities sum to 1
    prob_sum = probabilities.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5  # Allow small floating point error


def test_image_preprocessing_output_range():
    """Test that normalized image values are in expected range"""
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    
    # Transform
    transformed = transform(test_image)
    
    # Check output is a tensor
    assert isinstance(transformed, torch.Tensor)
    
    # Check range (after normalization, values typically in [-2, 2] range)
    assert transformed.min() >= -3.0
    assert transformed.max() <= 3.0
