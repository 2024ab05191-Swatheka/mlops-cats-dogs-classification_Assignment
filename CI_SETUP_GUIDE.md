# M3: CI Pipeline Setup Guide

## Overview
This guide explains how to set up the Continuous Integration (CI) pipeline for automated testing, building, and publishing of the Cats vs Dogs classifier.

## Prerequisites
- GitHub account
- GitHub repository for the project
- Docker Hub account (for Docker Hub publishing) OR GitHub Container Registry access

---

## 1. Automated Testing Setup

### Test Structure
```
tests/
├── __init__.py
├── test_preprocessing.py    # Tests for data preprocessing
└── test_inference.py         # Tests for model inference
```

### Running Tests Locally
```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Test Coverage
- **Preprocessing Tests**: Image resize, normalization, tensor conversion, augmentation
- **Inference Tests**: Model output shape, probability validation, batch processing

---

## 2. GitHub Actions CI Pipeline

### Workflow File Location
`.github/workflows/ci.yml`

### Pipeline Stages

#### Stage 1: Test
- Checks out code
- Sets up Python environment
- Installs dependencies
- Runs pytest with coverage
- Uploads coverage reports

#### Stage 2: Build
- Builds Docker image
- Tests the built image
- Uses caching for faster builds

#### Stage 3: Publish
- Publishes to GitHub Container Registry (GHCR)
- OR publishes to Docker Hub
- Only runs on main branch pushes
- Tags images with version, SHA, and 'latest'

### Triggering the Pipeline
The pipeline automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` branch

---

## 3. GitHub Container Registry Setup

### Enable GHCR for Your Repository
1. Go to your GitHub repository
2. Settings → Actions → General
3. Workflow permissions: Select "Read and write permissions"
4. Save changes

### Pulling Published Images
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull image
docker pull ghcr.io/USERNAME/REPO/cats-dogs-classifier:latest

# Run container
docker run -p 8000:8000 ghcr.io/USERNAME/REPO/cats-dogs-classifier:latest
```

---

## 4. Docker Hub Publishing Setup (Alternative)

### Create Docker Hub Access Token
1. Login to Docker Hub
2. Account Settings → Security → New Access Token
3. Name: `github-actions`
4. Copy the token

### Add Secrets to GitHub
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Add the following secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: The access token you created

### Pulling from Docker Hub
```bash
# Pull image
docker pull USERNAME/cats-dogs-classifier:latest

# Run container
docker run -p 8000:8000 USERNAME/cats-dogs-classifier:latest
```

---

## 5. Workflow Configuration

### Environment Variables (in ci.yml)
```yaml
env:
  DOCKER_IMAGE_NAME: cats-dogs-classifier
  DOCKER_REGISTRY: ghcr.io
  PYTHON_VERSION: "3.10"
```

### Customization
- Change `DOCKER_IMAGE_NAME` to your preferred image name
- Modify `PYTHON_VERSION` if needed
- Adjust test commands in the `test` job

---

## 6. Monitoring CI Pipeline

### Viewing Workflow Runs
1. Go to your GitHub repository
2. Click "Actions" tab
3. View all workflow runs, logs, and artifacts

### Status Badges
Add to your README.md:
```markdown
![CI Pipeline](https://github.com/USERNAME/REPO/workflows/CI%20Pipeline%20-%20Build,%20Test%20&%20Publish/badge.svg)
```

---

## 7. Local Testing Before Push

### Run tests locally
```bash
pytest tests/ -v
```

### Build Docker image locally
```bash
docker build -t cats-dogs-classifier:local .
```

### Test the image
```bash
docker run -p 8000:8000 cats-dogs-classifier:local
curl http://localhost:8000/health
```

---

## 8. Troubleshooting

### Tests Failing
- Check test logs in Actions tab
- Run tests locally: `pytest tests/ -v`
- Verify all dependencies in requirements.txt

### Docker Build Failing
- Check Dockerfile syntax
- Verify all files are committed (app.py, models/, requirements.txt)
- Test build locally: `docker build -t test .`

### Publishing Failing
- Verify GitHub token permissions (Settings → Actions)
- For Docker Hub: Check secrets are correctly set
- Ensure you're pushing to main branch

---

## 9. CI Pipeline Benefits

✓ **Automated Testing**: Every commit is tested automatically
✓ **Early Bug Detection**: Catch issues before deployment
✓ **Reproducible Builds**: Docker ensures consistency
✓ **Automated Deployment**: Push to registry on successful builds
✓ **Version Control**: Images tagged with git SHA and versions
✓ **Team Collaboration**: Clear status of all changes

---

## 10. Next Steps

1. Push code to GitHub repository
2. Verify CI pipeline runs successfully
3. Check published images in registry
4. Pull and test published images
5. Add status badges to README.md

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [pytest Documentation](https://docs.pytest.org/)
