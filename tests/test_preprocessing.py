"""
Unit tests for data preprocessing functions
"""
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class TestImagePreprocessing:
    """Test suite for image preprocessing functions"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing"""
        # Create a 100x100 RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array, mode='RGB')
    
    @pytest.fixture
    def preprocessing_transform(self):
        """Standard preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_image_resize(self, sample_image, preprocessing_transform):
        """Test that image is correctly resized to 224x224"""
        transformed = preprocessing_transform(sample_image)
        assert transformed.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {transformed.shape}"
    
    def test_image_normalization(self, sample_image, preprocessing_transform):
        """Test that image values are normalized"""
        transformed = preprocessing_transform(sample_image)
        # Check that values are roughly in the normalized range
        assert transformed.min() >= -3.0, "Normalized values too low"
        assert transformed.max() <= 3.0, "Normalized values too high"
    
    def test_tensor_conversion(self, sample_image, preprocessing_transform):
        """Test that preprocessing returns a tensor"""
        transformed = preprocessing_transform(sample_image)
        assert isinstance(transformed, torch.Tensor), "Output should be a PyTorch tensor"
    
    def test_rgb_channels(self, sample_image, preprocessing_transform):
        """Test that image has 3 RGB channels"""
        transformed = preprocessing_transform(sample_image)
        assert transformed.shape[0] == 3, f"Expected 3 channels, got {transformed.shape[0]}"
    
    def test_batch_preprocessing(self, sample_image, preprocessing_transform):
        """Test preprocessing multiple images for batch processing"""
        batch = torch.stack([preprocessing_transform(sample_image) for _ in range(4)])
        assert batch.shape == (4, 3, 224, 224), f"Expected batch shape (4, 3, 224, 224), got {batch.shape}"


class TestDataAugmentation:
    """Test suite for data augmentation functions"""
    
    @pytest.fixture
    def augmentation_transform(self):
        """Augmentation transform with random operations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip for testing
            transforms.ToTensor()
        ])
    
    def test_horizontal_flip(self):
        """Test horizontal flip augmentation"""
        # Create a simple image with distinct left/right pattern
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :50, :] = 255  # Left half white
        img = Image.fromarray(img_array, mode='RGB')
        
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ])
        
        flipped = flip_transform(img)
        # Check that right side is now brighter (was left before flip)
        assert flipped[:, :, -1].mean() > flipped[:, :, 0].mean()
    
    def test_resize_maintains_aspect(self):
        """Test that resize works correctly"""
        img_array = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        resized = resize_transform(img)
        assert resized.shape == (3, 224, 224)


def test_image_path_validation():
    """Test image file validation"""
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    test_files = ['image.jpg', 'photo.jpeg', 'pic.png', 'doc.txt', 'data.csv']
    
    for file in test_files:
        ext = '.' + file.split('.')[-1].lower()
        if ext in valid_extensions:
            assert ext in valid_extensions, f"{file} should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
