# Team Setup Guide

## For Team Members - Quick Start

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd mlops_project
```

### Step 2: Download Dataset
Download the Kaggle Cats and Dogs dataset:
- **Link**: https://www.kaggle.com/c/dogs-vs-cats/data
- Extract to your preferred location (e.g., `D:\datasets\PetImages`)

### Step 3: Configure Local Paths

**Option A: Using local_config.py (Recommended)**
```bash
# Copy the example file
cp local_config.example.py local_config.py

# Edit local_config.py and update RAW_DATA_PATH to your dataset location
```

**Option B: Using Environment Variables**
```bash
# Windows (PowerShell)
$env:DATASET_PATH = "C:\path\to\your\PetImages"

# Linux/Mac
export DATASET_PATH="/path/to/your/PetImages"
```

**Option C: Using DVC (Best for Team Collaboration)**
```bash
# Pull the dataset from remote storage
dvc pull

# Dataset will be automatically downloaded to the correct location
```

### Step 4: Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Initialize Git/DVC (if not already done)
```bash
git init
dvc init
```

### Step 6: Run the Notebook
```bash
jupyter notebook mlops_cats_dogs_pipeline.ipynb
```

## Using DVC for Team Data Sharing

### Why DVC?
- **No path issues**: Everyone gets the data in the same structure
- **Version control**: Track dataset changes
- **Remote storage**: Share large files without Git

### Setup DVC Remote (Team Lead)
```bash
# Option 1: Google Drive
dvc remote add -d myremote gdrive://FOLDER_ID
dvc push

# Option 2: AWS S3
dvc remote add -d s3remote s3://mybucket/dvc-storage
dvc push

# Option 3: Shared Network Drive
dvc remote add -d shared /mnt/shared/dvc-storage
dvc push
```

### Pull Data (Team Members)
```bash
# Configure remote (one-time)
dvc remote list  # See configured remotes

# Pull data
dvc pull

# Data is now in the correct location for everyone!
```

## File Structure
```
mlops_project/
├── mlops_cats_dogs_pipeline.ipynb  # Main notebook
├── config.py                        # Default config
├── local_config.py                  # Your local paths (gitignored)
├── local_config.example.py          # Template for team members
├── requirements.txt                 # Dependencies
├── .gitignore                       # Ignores local_config.py
├── .dvc/                           # DVC configuration
├── PetImages.dvc                   # DVC dataset pointer
├── data/                           # Processed data
├── models/                         # Saved models
└── experiments/                    # MLflow tracking
```

## Common Issues

### Issue: "Dataset not found"
**Solution**: Check your `local_config.py` or `DATASET_PATH` environment variable

### Issue: "Path conflicts between team members"
**Solution**: Use `local_config.py` (gitignored) instead of hardcoding paths

### Issue: "Can't download dataset"
**Solution**: 
1. Ask team lead for DVC remote access
2. Run `dvc pull`
3. Or download manually from Kaggle

### Issue: "Different Python versions"
**Solution**: Document required version in `README.md`, use virtual environments

## Best Practices for Teams

1. **Never commit local_config.py** - It's gitignored for a reason
2. **Use DVC for dataset sharing** - Don't share large files via Git
3. **Document your environment** - Add Python version, GPU specs to README
4. **Use relative paths** - They work for everyone
5. **Communicate changes** - Update TEAM_SETUP.md when workflow changes

## Contact
- **Team Lead**: [Your Name]
- **Slack Channel**: #mlops-cats-dogs
- **Issues**: Use GitHub Issues for bugs and questions
