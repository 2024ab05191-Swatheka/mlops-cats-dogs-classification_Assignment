# PowerShell script to test the Cats vs Dogs Classifier API
# Run this AFTER the Docker container is running

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Cats vs Dogs Classifier API - Test Script" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Check if Docker is installed
Write-Host "[1/6] Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    Write-Host "✓ Docker installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not installed!" -ForegroundColor Red
    Write-Host "`nPlease install Docker Desktop from:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop/`n" -ForegroundColor Cyan
    exit 1
}

# Check if container is running
Write-Host "`n[2/6] Checking if container is running..." -ForegroundColor Yellow
$containerRunning = docker ps --filter "name=cats-dogs-api" --format "{{.Names}}" 2>$null
if ($containerRunning -eq "cats-dogs-api") {
    Write-Host "✓ Container 'cats-dogs-api' is running" -ForegroundColor Green
} else {
    Write-Host "✗ Container is not running!" -ForegroundColor Red
    Write-Host "`nStarting container..." -ForegroundColor Yellow
    docker run -d -p 8000:8000 --name cats-dogs-api cats-dogs-classifier:v1.0 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Container started successfully" -ForegroundColor Green
        Start-Sleep -Seconds 3
    } else {
        Write-Host "✗ Failed to start container" -ForegroundColor Red
        Write-Host "Please build the image first: docker build -t cats-dogs-classifier:v1.0 ." -ForegroundColor Yellow
        exit 1
    }
}

# Test 1: Health Check
Write-Host "`n[3/6] Testing Health Check endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -ErrorAction Stop
    Write-Host "✓ Health Check passed" -ForegroundColor Green
    Write-Host "  Response: $($healthResponse | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Health Check failed" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
}

# Test 2: Root Endpoint
Write-Host "`n[4/6] Testing Root endpoint..." -ForegroundColor Yellow
try {
    $rootResponse = Invoke-RestMethod -Uri "http://localhost:8000/" -Method GET -ErrorAction Stop
    Write-Host "✓ Root endpoint passed" -ForegroundColor Green
    Write-Host "  API Version: $($rootResponse.version)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Root endpoint failed" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
}

# Prepare test image
Write-Host "`n[5/6] Preparing test image..." -ForegroundColor Yellow
$testImagePath = "test_cat.jpg"
if (-not (Test-Path $testImagePath)) {
    if (Test-Path "data\raw\PetImages\Cat\0.jpg") {
        Copy-Item "data\raw\PetImages\Cat\0.jpg" -Destination $testImagePath
        Write-Host "✓ Test image copied: $testImagePath" -ForegroundColor Green
    } else {
        Write-Host "✗ Test image not found in dataset" -ForegroundColor Red
        Write-Host "  Please ensure dataset is in data/raw/PetImages/" -ForegroundColor Yellow
    }
} else {
    Write-Host "✓ Test image already exists: $testImagePath" -ForegroundColor Green
}

# Test 3: Prediction Endpoint
Write-Host "`n[6/6] Testing Prediction endpoint..." -ForegroundColor Yellow
if (Test-Path $testImagePath) {
    try {
        # Prepare multipart form data
        $filePath = Resolve-Path $testImagePath
        $fileBytes = [System.IO.File]::ReadAllBytes($filePath)
        $fileName = Split-Path $filePath -Leaf
        $boundary = [System.Guid]::NewGuid().ToString()
        
        $LF = "`r`n"
        $bodyLines = (
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
            "Content-Type: image/jpeg$LF",
            [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
            "--$boundary--$LF"
        ) -join $LF
        
        $contentType = "multipart/form-data; boundary=$boundary"
        
        # Make prediction request
        $predictionResponse = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $bodyLines -ContentType $contentType -ErrorAction Stop
        
        Write-Host "✓ Prediction successful!" -ForegroundColor Green
        Write-Host "`n  Results:" -ForegroundColor Cyan
        Write-Host "    Predicted Class: $($predictionResponse.predicted_class)" -ForegroundColor White
        Write-Host "    Confidence: $([math]::Round($predictionResponse.confidence * 100, 2))%" -ForegroundColor White
        Write-Host "    Probabilities:" -ForegroundColor White
        Write-Host "      Cat: $([math]::Round($predictionResponse.probabilities.Cat * 100, 2))%" -ForegroundColor White
        Write-Host "      Dog: $([math]::Round($predictionResponse.probabilities.Dog * 100, 2))%" -ForegroundColor White
        
    } catch {
        Write-Host "✗ Prediction failed" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⊘ Skipping prediction test (no test image)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`nAll API endpoints are ready for testing!" -ForegroundColor Green
Write-Host "`nYou can also access:" -ForegroundColor Yellow
Write-Host "  • Swagger UI:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  • ReDoc:       http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host "  • Health:      http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "`nTo stop the container:" -ForegroundColor Yellow
Write-Host "  docker stop cats-dogs-api" -ForegroundColor Gray
Write-Host "`n" -ForegroundColor White
