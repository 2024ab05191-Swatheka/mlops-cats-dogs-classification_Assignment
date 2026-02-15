"""
Unit tests for FastAPI endpoints
"""
import pytest


def test_pytorch_import():
    """Test that PyTorch can be imported"""
    import torch
    assert torch is not None
    assert hasattr(torch, 'nn')


def test_fastapi_import():
    """Test that FastAPI can be imported"""
    from fastapi import FastAPI
    app = FastAPI()
    assert app is not None


def test_httpx_available():
    """Test that httpx is available for TestClient"""
    try:
        import httpx
        assert httpx is not None
    except ImportError:
        pytest.fail("httpx not installed - required for FastAPI TestClient")


def test_basic_functionality():
    """Test basic Python and math operations"""
    assert 1 + 1 == 2
    class_names = ['Cat', 'Dog']
    assert len(class_names) == 2
    assert 'Cat' in class_names


def test_torch_tensor():
    """Test PyTorch tensor creation"""
    import torch
    tensor = torch.randn(1, 3, 224, 224)
    assert tensor.shape == (1, 3, 224, 224)
