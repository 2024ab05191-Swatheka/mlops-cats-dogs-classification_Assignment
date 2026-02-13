# DVC Setup Guide

## What is DVC?
DVC (Data Version Control) is a version control system for ML projects that handles large datasets and model files. It works alongside Git to track data, models, and experiments.

## Installation

```bash
pip install dvc
```

## Initial Setup

### 1. Initialize DVC in your project
```bash
cd C:\Users\swath\dataset\mlops_project
dvc init
```

This creates:
- `.dvc/` directory
- `.dvcignore` file
- Updates `.gitignore`

### 2. Track the Dataset

Since your dataset is at `C:\Users\swath\dataset\archive (2)\PetImages`:

```bash
# Navigate to project root
cd C:\Users\swath\dataset\mlops_project

# Add the dataset (use relative or absolute path)
dvc add "..\archive (2)\PetImages"
```

This creates:
- `PetImages.dvc` - metadata file (this goes to Git)
- Updates `.gitignore` to exclude the actual data

### 3. Commit to Git

```bash
# Initialize Git if not done
git init

# Add DVC files
git add .dvc .gitignore PetImages.dvc

# Commit
git commit -m "Initialize DVC and track dataset"
```

## Setup Remote Storage (Optional but Recommended)

### Option 1: Local Remote (External Drive)
```bash
# Add local remote storage
dvc remote add -d myremote D:\dvc-storage

# Configure
dvc remote modify myremote url D:\dvc-storage

# Push data
dvc push
```

### Option 2: Cloud Storage (S3)
```bash
# Configure AWS S3
dvc remote add -d s3remote s3://my-bucket/dvc-storage

# Set credentials
dvc remote modify s3remote access_key_id YOUR_ACCESS_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET_KEY

# Push
dvc push
```

### Option 3: Azure Blob Storage
```bash
dvc remote add -d azure azure://mycontainer/path
dvc remote modify azure account_name YOUR_ACCOUNT
dvc remote modify azure account_key YOUR_KEY
dvc push
```

### Option 4: Google Drive
```bash
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
dvc push
```

## Common DVC Commands

### Track New Data/Models
```bash
# Track a file or directory
dvc add data/new_dataset.csv
dvc add models/model_v2.pt

# Commit to Git
git add data/new_dataset.csv.dvc models/model_v2.pt.dvc
git commit -m "Add new dataset and model"
```

### Pull Data (after cloning repo)
```bash
# Pull all tracked data
dvc pull

# Pull specific file
dvc pull data/dataset.csv.dvc
```

### Push Data to Remote
```bash
# Push all tracked data
dvc push

# Push specific file
dvc push models/model.pt.dvc
```

### Check Status
```bash
# Check DVC status
dvc status

# Check what's tracked
dvc list . --dvc-only
```

### Update Tracked Data
```bash
# If data changed
dvc add data/dataset.csv

# Commit changes
git add data/dataset.csv.dvc
git commit -m "Update dataset"
```

## Project-Specific Setup

### Track PetImages Dataset
```bash
# If not already done
cd C:\Users\swath\dataset\mlops_project
dvc add "..\archive (2)\PetImages"
git add "PetImages.dvc" .gitignore
git commit -m "Track PetImages dataset with DVC"
```

### Track Trained Models
```bash
# Track models directory
dvc add models/

# Or track specific models
dvc add models/best_model.pt
dvc add models/baseline_cnn_final.pt

# Commit
git add models.dvc .gitignore
git commit -m "Track trained models with DVC"
```

### Track Experiment Artifacts
```bash
dvc add experiments/
git add experiments.dvc
git commit -m "Track experiment artifacts"
```

## DVC Pipeline (Advanced)

Create a `dvc.yaml` for reproducible pipelines:

```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - scripts/preprocess.py
      - data/raw/
    outs:
      - data/processed/
    
  train:
    cmd: python scripts/train.py
    deps:
      - scripts/train.py
      - data/processed/
    params:
      - train.learning_rate
      - train.batch_size
    outs:
      - models/model.pt
    metrics:
      - metrics.json:
          cache: false
```

Run pipeline:
```bash
dvc repro
```

## Troubleshooting

### Issue: "DVC is not initialized"
```bash
dvc init
git add .dvc
git commit -m "Initialize DVC"
```

### Issue: "Cannot find .dvc file"
```bash
# Re-add the file
dvc add path/to/file
git add path/to/file.dvc
```

### Issue: "Remote storage not configured"
```bash
# Check remote
dvc remote list

# Add remote if missing
dvc remote add -d myremote /path/to/storage
```

### Issue: "Data not syncing"
```bash
# Force update
dvc fetch --all-commits
dvc checkout --force
```

## Best Practices

1. **Track Large Files**: Use DVC for files > 10MB
2. **Don't Track Code**: Use Git for .py, .ipynb files
3. **Use Remote Storage**: Always configure a remote
4. **Commit .dvc files**: These go to Git, not actual data
5. **Ignore Data in Git**: Ensure `.gitignore` excludes data directories
6. **Version Metadata**: Track `dataset_metadata.json` with Git
7. **Tag Versions**: Use Git tags for dataset versions

## Workflow Example

```bash
# 1. Make changes to dataset
python scripts/clean_data.py

# 2. Track changes with DVC
dvc add data/cleaned_dataset.csv

# 3. Commit metadata to Git
git add data/cleaned_dataset.csv.dvc
git commit -m "Clean dataset v2"

# 4. Push data to remote
dvc push

# 5. Push code to Git
git push origin main

# On another machine:
# 1. Clone repo
git clone <repo-url>

# 2. Pull data
dvc pull
```

## Resources

- Official Docs: https://dvc.org/doc
- Get Started: https://dvc.org/doc/start
- Remote Storage: https://dvc.org/doc/command-reference/remote
- Pipelines: https://dvc.org/doc/start/data-pipelines

---

**Note**: DVC setup is optional but highly recommended for tracking large datasets and models in production environments.
