# 🚀 How to Push Your Project to GitHub

## Project Structure (Organized)

Your project is now organized like a professional MLOps repository:

```
mlops_project/
├── 📁 .github/workflows/     # CI/CD pipeline
│   └── ci.yml
├── 📁 app/                   # FastAPI application
│   ├── app.py
│   └── config.py
├── 📁 data/                  # Dataset
│   ├── raw/PetImages.dvc
│   └── dataset_metadata.json
├── 📁 docker/                # Docker compose configs
│   └── docker-compose.yml
├── 📁 k8s/                   # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── 📁 models/                # Trained models
│   └── best_model.pt
├── 📁 notebooks/             # Jupyter notebooks
│   └── mlops_cats_dogs_pipeline.ipynb
├── 📁 src/                   # Source code (if needed)
├── 📁 tests/                 # Test files
├── 📄 Dockerfile            # Container definition
├── 📄 requirements.txt      # Python dependencies
├── 📄 README.md             # Project documentation
├── 📄 .gitignore            # Git ignore rules
└── 📄 pytest.ini            # Test configuration
```

---

## Step 1: Update Dockerfile for New Structure

Since we moved `app.py`, update the Dockerfile:

```dockerfile
# Copy application code from app folder
COPY app/app.py app.py
COPY app/config.py config.py
COPY models/best_model.pt models/best_model.pt
```

---

## Step 2: Create/Update .gitignore

Make sure you're not pushing large files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Data (too large)
data/raw/PetImages/
*.jpg
*.png
*.jpeg

# Models (if too large, use Git LFS or DVC)
# models/*.pt

# MLflow
experiments/
mlruns/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# DVC
.dvc/cache
```

---

## Step 3: Initialize Git Repository (if not already done)

```powershell
cd c:\Users\swath\dataset\mlops_project

# Initialize git (skip if already done)
git init

# Check status
git status
```

---

## Step 4: Create GitHub Repository

### Option A: Using GitHub Website

1. Go to [github.com](https://github.com)
2. Click the **"+"** icon → **"New repository"**
3. Repository name: `mlops-cats-dogs-classification`
4. Description: "Complete MLOps pipeline for Cats vs Dogs image classification"
5. Select: **Public** (or Private if you prefer)
6. **Do NOT** initialize with README (you already have one)
7. Click **"Create repository"**

### Option B: Using GitHub CLI (if installed)

```powershell
gh repo create mlops-cats-dogs-classification --public --source=. --remote=origin
```

---

## Step 5: Connect Local Repo to GitHub

After creating the repo on GitHub, you'll see commands like these. Copy YOUR actual URL:

```powershell
cd c:\Users\swath\dataset\mlops_project

# Add remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/mlops-cats-dogs-classification.git

# Or if using SSH:
# git remote add origin git@github.com:USERNAME/mlops-cats-dogs-classification.git

# Verify remote
git remote -v
```

---

## Step 6: Stage All Files

```powershell
# Add all files
git add .

# Check what will be committed
git status
```

---

## Step 7: Commit Your Code

```powershell
git commit -m "Initial commit: Complete MLOps pipeline with M1-M5 implementation"
```

---

## Step 8: Push to GitHub

```powershell
# Push to main branch
git push -u origin main

# If your default branch is 'master':
# git push -u origin master

# Or if you need to set upstream:
git branch -M main
git push -u origin main
```

---

## Step 9: Verify on GitHub

1. Go to your GitHub repository URL
2. You should see all your folders and files!
3. Check that:
   - ✅ README.md is displayed on the homepage
   - ✅ All folders are visible (app/, notebooks/, k8s/, etc.)
   - ✅ CI/CD workflow is in .github/workflows/
   - ✅ Large files are not pushed (check .gitignore)

---

## Step 10: Set Up GitHub Actions (CI/CD)

Your CI/CD pipeline should automatically appear in the **"Actions"** tab after pushing.

If you push again:
```powershell
git add .
git commit -m "Update: Your message here"
git push
```

The GitHub Actions pipeline will automatically:
- ✅ Run tests
- ✅ Build Docker image
- ✅ Publish to container registry

---

## 🎯 Quick Command Sequence

If you already have a GitHub repo URL:

```powershell
cd c:\Users\swath\dataset\mlops_project

# Remove old remote if exists
git remote remove origin

# Add your new remote (REPLACE WITH YOUR URL)
git remote add origin https://github.com/YOUR_USERNAME/mlops-cats-dogs-classification.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Complete MLOps pipeline"

# Push
git branch -M main
git push -u origin main
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "Large files" error

**Solution:** Models and datasets are too large. Either:
1. Use Git LFS: `git lfs track "*.pt"`
2. Or track with DVC (already done): `dvc push`
3. Or add to .gitignore and upload separately

### Issue 2: "Permission denied"

**Solution:** Set up authentication:
```powershell
# Use GitHub CLI
gh auth login

# Or use Personal Access Token
# Settings → Developer settings → Personal access tokens → Generate new token
```

### Issue 3: "Failed to push"

**Solution:** Pull first if repo has content:
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Issue 4: "Repository not found"

**Solution:** Check the URL is correct:
```powershell
git remote -v
# Update if wrong:
git remote set-url origin https://github.com/CORRECT_USERNAME/CORRECT_REPO.git
```

---

## 📝 After Pushing to GitHub

### Update Your Dockerfile

Since files moved, update the Dockerfile copy commands to reflect new structure. Currently at root, but should reference new paths.

### Update README with GitHub Badge

Add this to your README.md:

```markdown
# MLOps Cats vs Dogs Classification

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/mlops-cats-dogs-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mlops-cats-dogs-classification/actions)

Complete MLOps pipeline for image classification.
```

### Share Your Repository

Your GitHub URL will be:
```
https://github.com/YOUR_USERNAME/mlops-cats-dogs-classification
```

Share this link in your assignment submission!

---

## ✅ Final Checklist

Before submitting:

- [ ] All code pushed to GitHub
- [ ] Repository is public (or accessible to evaluators)
- [ ] README.md is comprehensive
- [ ] CI/CD pipeline is visible in Actions tab
- [ ] .gitignore excludes large files
- [ ] Project structure is organized
- [ ] All folders are properly named

---

## 🎉 You're Done!

Your project is now:
- ✅ Professionally organized
- ✅ Version controlled with Git
- ✅ Hosted on GitHub
- ✅ Ready for CI/CD automation
- ✅ Ready for team collaboration

**GitHub Repository URL to submit:**
```
https://github.com/YOUR_USERNAME/mlops-cats-dogs-classification
```

Good luck with your submission! 🚀
