# M2: Docker Setup and Testing Guide

## Current Status
✅ All required files are present:
- `Dockerfile` - Container configuration
- `app.py` - FastAPI inference service
- `models/best_model.pt` - Trained model
- `requirements.txt` - Dependencies

❌ Docker is not installed on your system

---

## Step 1: Install Docker

### Windows Installation

1. **Download Docker Desktop**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download Docker Desktop for Windows
   - Requires Windows 10/11 (64-bit)

2. **Install Docker Desktop**
   - Run the installer
   - Follow the installation wizard
   - Enable WSL 2 if prompted (recommended)
   - Restart your computer

3. **Verify Installation**
   ```powershell
   docker --version
   docker-compose --version
   ```

4. **Start Docker Desktop**
   - Launch Docker Desktop from Start Menu
   - Wait for Docker to start (icon in system tray)

---

## Step 2: Build Docker Image

Once Docker is installed and running:

```powershell
# Navigate to project directory
cd C:\Users\swath\dataset\mlops_project

# Build the Docker image
docker build -t cats-dogs-classifier:v1.0 .
```

**Expected Output:**
```
[+] Building 120.5s (10/10) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 450B
 => [internal] load .dockerignore
 => [internal] load metadata for docker.io/library/python:3.10-slim
 => [1/5] FROM docker.io/library/python:3.10-slim
 => [internal] load build context
 => [2/5] WORKDIR /app
 => [3/5] COPY requirements.txt .
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt
 => [5/5] COPY app.py .
 => [6/5] COPY models/best_model.pt models/best_model.pt
 => exporting to image
 => => naming to docker.io/library/cats-dogs-classifier:v1.0

Successfully built <image_id>
Successfully tagged cats-dogs-classifier:v1.0
```

**Verify Image:**
```powershell
docker images | grep cats-dogs-classifier
```

---

## Step 3: Run Docker Container

```powershell
# Run container in detached mode
docker run -d -p 8000:8000 --name cats-dogs-api cats-dogs-classifier:v1.0

# Check if container is running
docker ps

# View container logs
docker logs cats-dogs-api
```

**Expected Output (docker ps):**
```
CONTAINER ID   IMAGE                         COMMAND                  STATUS          PORTS
abc123def456   cats-dogs-classifier:v1.0     "uvicorn app:app --h…"   Up 10 seconds   0.0.0.0:8000->8000/tcp
```

---

## Step 4: Test API Endpoints

### Test 1: Health Check

```powershell
# PowerShell (using Invoke-WebRequest)
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET | Select-Object -Expand Content

# Or using curl (if installed)
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "service": "Cats vs Dogs Classifier"
}
```

### Test 2: Root Endpoint

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET | Select-Object -Expand Content
```

**Expected Response:**
```json
{
  "message": "Cats vs Dogs Classifier API",
  "version": "1.0",
  "endpoints": {
    "/health": "Health check",
    "/predict": "Prediction endpoint (POST image file)"
  }
}
```

### Test 3: Prediction Endpoint

First, get a test image from the dataset:

```powershell
# Copy a test image
Copy-Item "data\raw\PetImages\Cat\0.jpg" -Destination "test_cat.jpg"
```

Then test prediction:

```powershell
# PowerShell method
$filePath = "test_cat.jpg"
$uri = "http://localhost:8000/predict"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$boundary = [System.Guid]::NewGuid().ToString()
$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"test_cat.jpg`"",
    "Content-Type: image/jpeg",
    "",
    [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
    "--$boundary--"
)
$body = $bodyLines -join "`r`n"
$contentType = "multipart/form-data; boundary=$boundary"
Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType $contentType
```

**Or using curl (if available):**
```powershell
curl -X POST "http://localhost:8000/predict" -F "file=@test_cat.jpg"
```

**Expected Response:**
```json
{
  "predicted_class": "Cat",
  "confidence": 0.9876,
  "probabilities": {
    "Cat": 0.9876,
    "Dog": 0.0124
  }
}
```

---

## Step 5: Container Management

### View Logs
```powershell
docker logs cats-dogs-api

# Follow logs in real-time
docker logs -f cats-dogs-api
```

### Stop Container
```powershell
docker stop cats-dogs-api
```

### Start Container Again
```powershell
docker start cats-dogs-api
```

### Restart Container
```powershell
docker restart cats-dogs-api
```

### Remove Container
```powershell
docker stop cats-dogs-api
docker rm cats-dogs-api
```

### Remove Image
```powershell
docker rmi cats-dogs-classifier:v1.0
```

---

## Step 6: Access Interactive Documentation

Once the container is running, you can access the FastAPI interactive documentation:

**Swagger UI:**
- URL: http://localhost:8000/docs
- Interactive API documentation with "Try it out" functionality

**ReDoc:**
- URL: http://localhost:8000/redoc
- Alternative documentation format

---

## Troubleshooting

### Issue: Port 8000 Already in Use

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or run container on a different port
docker run -d -p 8001:8000 --name cats-dogs-api cats-dogs-classifier:v1.0
```

### Issue: Container Exits Immediately

```powershell
# Check logs for errors
docker logs cats-dogs-api

# Run container interactively to see errors
docker run -it --rm -p 8000:8000 cats-dogs-classifier:v1.0
```

### Issue: Model File Not Found

```powershell
# Verify model file exists before building
Test-Path models/best_model.pt

# If false, run the training notebook to generate the model
```

### Issue: Out of Memory

```powershell
# Increase Docker memory limit in Docker Desktop settings
# Settings > Resources > Memory
# Recommended: At least 4GB
```

---

## Alternative: Run Without Docker

If you prefer to test without Docker:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate

# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart

# Run the API directly
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Or
uvicorn app:app --reload
```

Then test the same way as described in Step 4.

---

## Assignment Verification Checklist

For M2 submission, ensure you have:

- [x] ✅ Task 1: Inference Service
  - [x] REST API with FastAPI
  - [x] `/health` endpoint implemented
  - [x] `/predict` endpoint implemented
  - [x] Returns class probabilities and label

- [x] ✅ Task 2: Environment Specification
  - [x] `requirements.txt` created
  - [x] All dependencies version-pinned

- [x] ✅ Task 3: Containerization
  - [x] `Dockerfile` created
  - [ ] ⏳ Image built locally (requires Docker)
  - [ ] ⏳ Container running (requires Docker)
  - [ ] ⏳ Verified predictions via curl/Postman (requires Docker)

---

## Documentation Files

All documentation has been created:
- `Dockerfile` - Container configuration
- `app.py` - FastAPI service
- `requirements.txt` - Dependencies
- `DOCKER_INSTRUCTIONS.txt` - Build & run commands
- `API_TESTING.txt` - Testing methods
- `M2_DOCKER_SETUP.md` - This comprehensive guide

---

## Next Steps

1. **Install Docker Desktop** (if not already installed)
2. **Build the Docker image** using the command above
3. **Run the container** and verify it starts successfully
4. **Test all endpoints** using curl or Postman
5. **Take screenshots** of successful API responses for assignment submission
6. **Document the results** in your assignment report

---

**Note:** All code files (app.py, Dockerfile, requirements.txt) are already created and ready to use. You only need to install Docker and run the commands!
