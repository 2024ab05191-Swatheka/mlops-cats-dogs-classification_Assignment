# DVC Remote Storage Setup

## Overview
Setting up a DVC remote allows your team to share the dataset without manually copying files. Team members can simply run `dvc pull` to download the data.

## Current Status
✅ Dataset tracked locally with DVC (`data/raw/PetImages.dvc`)  
❌ No remote storage configured yet

---

## Option 1: Google Drive (Recommended for Small Teams)

### Setup Steps

1. **Create a Google Drive folder** for the dataset
   - Go to [Google Drive](https://drive.google.com)
   - Create a new folder: `mlops-cats-dogs-dataset`
   - Right-click → Share → Get link → Copy the folder ID
   - Folder ID is the part after `/folders/` in the URL
   - Example: `https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0` → ID is `1a2b3c4d5e6f7g8h9i0`

2. **Configure DVC remote**
   ```bash
   cd mlops_project
   dvc remote add -d storage gdrive://1a2b3c4d5e6f7g8h9i0
   git add .dvc/config
   git commit -m "Add Google Drive as DVC remote"
   ```

3. **Push dataset to remote**
   ```bash
   dvc push
   ```
   - First time: Browser will open for Google authentication
   - Choose the Google account that owns the folder
   - Grant permissions

4. **Team members can now pull**
   ```bash
   git clone <repository-url>
   cd mlops_project
   dvc pull
   ```

---

## Option 2: AWS S3 (Recommended for Production)

### Prerequisites
- AWS account with S3 access
- AWS CLI installed and configured (`aws configure`)

### Setup Steps

1. **Create S3 bucket**
   ```bash
   aws s3 mb s3://mlops-cats-dogs-dataset
   ```

2. **Configure DVC remote**
   ```bash
   dvc remote add -d storage s3://mlops-cats-dogs-dataset/data
   git add .dvc/config
   git commit -m "Add S3 as DVC remote"
   ```

3. **Push dataset**
   ```bash
   dvc push
   ```

4. **Team setup** (each member needs AWS credentials)
   ```bash
   aws configure
   dvc pull
   ```

---

## Option 3: Azure Blob Storage

### Prerequisites
- Azure account with Storage account
- Azure CLI installed (`az login`)

### Setup Steps

1. **Create storage account and container**
   ```bash
   az storage account create -n mlopsstorage -g mlops-resource-group
   az storage container create -n dataset --account-name mlopsstorage
   ```

2. **Get connection string**
   ```bash
   az storage account show-connection-string -n mlopsstorage
   ```

3. **Configure DVC remote**
   ```bash
   dvc remote add -d storage azure://dataset
   dvc remote modify storage connection_string "<connection-string>"
   git add .dvc/config
   git commit -m "Add Azure as DVC remote"
   ```

4. **Push dataset**
   ```bash
   dvc push
   ```

---

## Option 4: SSH/Local Server

### Setup (if you have a shared server)

```bash
dvc remote add -d storage ssh://user@server/path/to/dvc-storage
dvc remote modify storage password <password>
# OR use SSH key authentication
git add .dvc/config
git commit -m "Add SSH remote"
dvc push
```

---

## Option 5: Local/Network Shared Drive (Quick Start)

### For Testing or Local Teams

```bash
# Windows
dvc remote add -d storage \\network-drive\mlops\dvc-cache

# Linux/Mac
dvc remote add -d storage /mnt/shared/mlops/dvc-cache

git add .dvc/config
git commit -m "Add network drive as DVC remote"
dvc push
```

---

## Verification

### Check remote configuration
```bash
dvc remote list
dvc remote default
```

### Test push/pull
```bash
# Push data to remote
dvc push

# Simulate team member
rm -rf data/raw/PetImages
dvc pull  # Should download from remote
```

---

## Team Member Workflow

### First-time setup
```bash
# 1. Clone repository
git clone <repository-url>
cd mlops_project

# 2. Create Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull dataset with DVC
dvc pull

# 5. Run notebook
jupyter notebook mlops_cats_dogs_pipeline.ipynb
```

### Daily workflow
```bash
# Pull latest code and data
git pull
dvc pull

# After making changes
git add .
git commit -m "Your changes"
git push
```

---

## Cost Considerations

| Storage Option | Free Tier | Best For |
|---------------|-----------|----------|
| **Google Drive** | 15 GB free | Small teams, quick setup |
| **AWS S3** | 5 GB free (12 months) | Production, large datasets |
| **Azure Blob** | 5 GB free (12 months) | Enterprise, Azure ecosystem |
| **SSH Server** | Depends on server | Full control, privacy |
| **Local/Network** | Depends on drive | Testing, local teams |

---

## Recommended: Google Drive for This Project

**Dataset size**: ~809 MB (24,998 images)  
**Recommendation**: Google Drive (fits in free tier, easy authentication)

### Quick Setup Command
```bash
# Replace YOUR_FOLDER_ID with actual Google Drive folder ID
dvc remote add -d storage gdrive://YOUR_FOLDER_ID
git add .dvc/config
git commit -m "Add Google Drive as DVC remote"
dvc push
```

---

## Troubleshooting

### Issue: "Authentication failed"
- **Solution**: Run `dvc remote modify storage --local auth basic` and retry

### Issue: "Permission denied"
- **Solution**: Ensure shared folder has "Editor" access for all team members

### Issue: "Failed to push"
- **Solution**: Check internet connection and remote credentials

### Issue: Large transfer times
- **Solution**: Use cloud storage (S3/Azure) instead of Google Drive for datasets >1GB

---

## Next Steps

1. Choose a remote storage option above
2. Configure the remote with `dvc remote add`
3. Push dataset with `dvc push`
4. Share repository URL with team
5. Team members run `dvc pull` to get dataset

**Documentation**: https://dvc.org/doc/command-reference/remote
