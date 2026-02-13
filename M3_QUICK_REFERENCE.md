# M3: CI Pipeline - Quick Reference

## Quick Start

### 1. Run Tests Locally
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### 2. Push to GitHub
```bash
git add .
git commit -m "Add M3: CI pipeline with automated testing"
git push origin main
```

### 3. Monitor CI Pipeline
- Go to: GitHub repository → Actions tab
- View real-time workflow execution
- Check test results and build status

### 4. Pull Published Image
```bash
# From GitHub Container Registry
docker pull ghcr.io/USERNAME/REPO/cats-dogs-classifier:latest

# From Docker Hub (if configured)
docker pull USERNAME/cats-dogs-classifier:latest
```

---

## Test Structure

```
tests/
├── __init__.py
├── test_preprocessing.py    # 8 tests for data preprocessing
└── test_inference.py         # 12 tests for model inference
```

### Test Categories

**Preprocessing Tests:**
- Image resize validation
- Normalization checks
- Tensor conversion
- RGB channel verification
- Batch preprocessing
- Data augmentation (flips, rotations)

**Inference Tests:**
- Model output shape
- Gradient computation (eval mode)
- Softmax probability validation
- Prediction class range
- Batch inference
- Deterministic inference
- Confidence score validation
- Response format validation

---

## CI Pipeline Stages

### Stage 1: Test
- ✓ Checkout repository
- ✓ Setup Python 3.10
- ✓ Install dependencies with caching
- ✓ Run pytest with coverage
- ✓ Upload coverage reports

### Stage 2: Build
- ✓ Setup Docker Buildx
- ✓ Build Docker image
- ✓ Test built image
- ✓ Use layer caching for speed

### Stage 3: Publish (main branch only)
- ✓ Login to container registry
- ✓ Extract image metadata
- ✓ Tag image (latest, SHA, version)
- ✓ Push to registry

---

## Registry Setup

### GitHub Container Registry (Default)
1. Enable GHCR:
   - Settings → Actions → General
   - Workflow permissions: "Read and write permissions"

2. Images published to: `ghcr.io/USERNAME/REPO/cats-dogs-classifier`

### Docker Hub (Alternative)
1. Create access token at hub.docker.com
2. Add GitHub secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
3. Images published to: `USERNAME/cats-dogs-classifier`

---

## Image Tags

- `latest` - Most recent build from main
- `main-<sha>` - Specific commit from main
- `v1.0`, `v1.1` - Semantic versions (if configured)

---

## Troubleshooting

### Tests Failing Locally
```bash
# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with verbose output
pytest tests/ -vv
```

### CI Pipeline Failing
1. Check Actions tab for detailed logs
2. Verify all files are committed
3. Check requirements.txt includes all dependencies
4. Ensure Docker build works locally

### Publishing Failing
1. Verify GHCR permissions (Settings → Actions)
2. Check secrets are set (for Docker Hub)
3. Ensure pushing to main branch

---

## Files Created

```
mlops_project/
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_inference.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pytest.ini
├── CI_SETUP_GUIDE.md
├── REGISTRY_SETUP.txt
└── RUN_TESTS.txt
```

---

## CI Benefits

✅ **Automated Testing** - Every commit tested automatically
✅ **Early Detection** - Catch bugs before deployment
✅ **Reproducible Builds** - Docker ensures consistency
✅ **Version Control** - All images tagged and traceable
✅ **Team Collaboration** - Clear status of all changes
✅ **Automated Deployment** - Push to registry on success

---

## Next Steps

1. ✓ Tests created and passing locally
2. ✓ CI workflow configured
3. ⏳ Push to GitHub
4. ⏳ Monitor first pipeline run
5. ⏳ Pull and test published image
6. ⏳ Add CI badge to README

---

## Status Badge

Add to README.md:

```markdown
![CI Pipeline](https://github.com/USERNAME/REPO/workflows/CI%20Pipeline%20-%20Build,%20Test%20&%20Publish/badge.svg)
```

---

## Additional Resources

- Full guide: `CI_SETUP_GUIDE.md`
- Registry setup: `REGISTRY_SETUP.txt`
- Test instructions: `RUN_TESTS.txt`
- GitHub Actions docs: https://docs.github.com/en/actions
- pytest docs: https://docs.pytest.org/
