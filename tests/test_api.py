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


def test_root_endpoint():
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["version"] == "1.0"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert "service" in data


def test_metrics_endpoint():
    """Test metrics endpoint returns tracking data"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "metrics" in data
    assert "total_predictions" in data["metrics"]
    assert "cat_predictions" in data["metrics"]
    assert "dog_predictions" in data["metrics"]
    assert "health_checks" in data["metrics"]


def test_predict_endpoint_no_file():
    """Test predict endpoint without file returns error"""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error for missing file


def test_metrics_increment():
    """Test that metrics increment correctly"""
    # Get initial metrics
    response1 = client.get("/metrics")
    initial_health_checks = response1.json()["metrics"]["health_checks"]
    
    # Call health endpoint
    client.get("/health")
    
    # Check metrics incremented
    response2 = client.get("/metrics")
    new_health_checks = response2.json()["metrics"]["health_checks"]
    
    assert new_health_checks == initial_health_checks + 1
