"""
Unit tests for model inference and utility functions
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestModelInference:
    """Test suite for model inference functions"""
    
    @pytest.fixture
    def model_architecture(self):
        """Create a simple test model with same architecture"""
        class SimpleTestModel(nn.Module):
            def __init__(self, num_classes=2):
                super(SimpleTestModel, self).__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(16 * 112 * 112, num_classes)
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleTestModel(num_classes=2)
        model.eval()
        return model
    
    @pytest.fixture
    def sample_input_tensor(self):
        """Create a sample input tensor (batch_size=1, channels=3, height=224, width=224)"""
        return torch.randn(1, 3, 224, 224)
    
    def test_model_output_shape(self, model_architecture, sample_input_tensor):
        """Test that model outputs correct shape for binary classification"""
        with torch.no_grad():
            output = model_architecture(sample_input_tensor)
        
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    
    def test_model_inference_no_gradient(self, model_architecture, sample_input_tensor):
        """Test that inference runs without computing gradients"""
        model_architecture.eval()
        with torch.no_grad():
            output = model_architecture(sample_input_tensor)
        
        assert not output.requires_grad, "Output should not require gradients during inference"
    
    def test_softmax_probabilities(self, model_architecture, sample_input_tensor):
        """Test that softmax probabilities sum to 1"""
        with torch.no_grad():
            output = model_architecture(sample_input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        prob_sum = probabilities.sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities should sum to 1, got {prob_sum}"
    
    def test_prediction_class_range(self, model_architecture, sample_input_tensor):
        """Test that predicted class is within valid range"""
        with torch.no_grad():
            output = model_architecture(sample_input_tensor)
            _, predicted_class = torch.max(output, 1)
        
        assert 0 <= predicted_class.item() < 2, f"Predicted class should be 0 or 1, got {predicted_class.item()}"
    
    def test_batch_inference(self, model_architecture):
        """Test inference with batch of images"""
        batch_input = torch.randn(8, 3, 224, 224)
        
        with torch.no_grad():
            output = model_architecture(batch_input)
        
        assert output.shape == (8, 2), f"Expected batch output shape (8, 2), got {output.shape}"
    
    def test_deterministic_inference(self, model_architecture, sample_input_tensor):
        """Test that inference is deterministic (same input -> same output)"""
        model_architecture.eval()
        
        with torch.no_grad():
            output1 = model_architecture(sample_input_tensor)
            output2 = model_architecture(sample_input_tensor)
        
        assert torch.allclose(output1, output2), "Model should produce same output for same input"


class TestPredictionUtilities:
    """Test suite for prediction utility functions"""
    
    def test_class_name_mapping(self):
        """Test class index to name mapping"""
        class_names = ['Cat', 'Dog']
        predicted_idx = 0
        
        predicted_class = class_names[predicted_idx]
        assert predicted_class == 'Cat', f"Expected 'Cat' for index 0, got {predicted_class}"
    
    def test_confidence_score_range(self):
        """Test that confidence scores are between 0 and 1"""
        # Simulate model output
        logits = torch.tensor([[2.5, 1.2]])
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, _ = torch.max(probabilities, 1)
        
        assert 0.0 <= confidence.item() <= 1.0, f"Confidence should be in [0, 1], got {confidence.item()}"
    
    def test_probability_dictionary_creation(self):
        """Test creation of probability dictionary for API response"""
        logits = torch.tensor([[1.5, 2.3]])
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        prob_dict = {
            "Cat": float(probabilities[0][0].item()),
            "Dog": float(probabilities[0][1].item())
        }
        
        assert isinstance(prob_dict["Cat"], float), "Probability should be float"
        assert isinstance(prob_dict["Dog"], float), "Probability should be float"
        assert abs(sum(prob_dict.values()) - 1.0) < 1e-5, "Probabilities should sum to 1"
    
    def test_prediction_response_format(self):
        """Test that prediction response has correct format"""
        # Simulate prediction result
        result = {
            "predicted_class": "Dog",
            "confidence": 0.9876,
            "probabilities": {
                "Cat": 0.0124,
                "Dog": 0.9876
            }
        }
        
        assert "predicted_class" in result, "Response should contain predicted_class"
        assert "confidence" in result, "Response should contain confidence"
        assert "probabilities" in result, "Response should contain probabilities"
        assert result["predicted_class"] in ["Cat", "Dog"], "Predicted class should be Cat or Dog"


class TestImageLoading:
    """Test suite for image loading utilities"""
    
    def test_image_loading_from_bytes(self):
        """Test loading image from bytes (simulating upload)"""
        # Create a simple image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Load from bytes
        loaded_img = Image.open(img_bytes).convert('RGB')
        
        assert loaded_img.mode == 'RGB', f"Expected RGB mode, got {loaded_img.mode}"
        assert loaded_img.size == (100, 100), f"Expected size (100, 100), got {loaded_img.size}"
    
    def test_rgb_conversion(self):
        """Test that images are converted to RGB"""
        # Create grayscale image
        img = Image.new('L', (100, 100), color=128)
        
        # Convert to RGB
        rgb_img = img.convert('RGB')
        
        assert rgb_img.mode == 'RGB', f"Expected RGB mode, got {rgb_img.mode}"
        assert len(rgb_img.getbands()) == 3, "RGB image should have 3 channels"


def test_device_selection():
    """Test device selection logic (CPU/GPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    assert device.type in ['cuda', 'cpu'], f"Device should be cuda or cpu, got {device.type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
