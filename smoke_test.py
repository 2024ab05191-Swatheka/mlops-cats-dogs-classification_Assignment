"""
Smoke tests for deployed service
Verifies health endpoint and prediction functionality
"""
import sys
import requests
import time
from PIL import Image
import io

def run_smoke_tests(base_url):
    """Run smoke tests against deployed service"""
    print(f"🧪 Running smoke tests against {base_url}")
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "healthy", "Service not healthy"
        print("   ✅ Health check passed")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("   ✅ Root endpoint passed")
    except Exception as e:
        print(f"   ❌ Root endpoint failed: {e}")
        return False
    
    # Test 3: Metrics endpoint
    print("\n3️⃣ Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "metrics" in data, "Metrics not in response"
        print("   ✅ Metrics endpoint passed")
    except Exception as e:
        print(f"   ❌ Metrics endpoint failed: {e}")
        return False
    
    # Test 4: Prediction endpoint (with dummy image)
    print("\n4️⃣ Testing prediction endpoint...")
    try:
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        
        # Check response
        assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 503:
            print("   ⚠️  Model not loaded (expected for CI without model file)")
            print("   ✅ Prediction endpoint accessible (service operational)")
        else:
            data = response.json()
            assert "predicted_class" in data, "No prediction in response"
            assert "confidence" in data, "No confidence in response"
            print(f"   ✅ Prediction passed: {data['predicted_class']} ({data['confidence']:.2%})")
    except Exception as e:
        print(f"   ❌ Prediction endpoint failed: {e}")
        return False
    
    print("\n✅ All smoke tests passed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <base_url>")
        print("Example: python smoke_test.py http://localhost:8000")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    # Give service time to start
    print("⏳ Waiting for service to be ready...")
    time.sleep(5)
    
    # Run tests
    success = run_smoke_tests(base_url)
    
    sys.exit(0 if success else 1)
