# MLOps Pipeline: Cats vs Dogs Classification

GitHub Link: https://github.com/2024ab05191-Swatheka/mlops-cats-dogs-classification_Assignment.git

Google Drive : https://drive.google.com/drive/folders/1j3OqX4PV3L917TouYgIRxGTGIkZ4AXJX?usp=drive_link
Recording Link: https://drive.google.com/file/d/1Ep_N0dZ0B4ixNBunzVdKDQI_MUG8ehOa/view?usp=drive_link

## 🎯 Project Overview

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform. This project demonstrates best practices in ML model development, experiment tracking, versioning, containerization, and CI/CD deployment.

## 📊 Dataset

- **Source**: Kaggle Cats and Dogs Dataset
- **Total Samples**: ~24,000 images
- **Classes**: Cat (0), Dog (1)
- **Split**: 80% Train / 10% Validation / 10% Test
- **Format**: .jpg images
- **Target Size**: 224x224 RGB

## 🏗️ Architecture

**Model**: Baseline CNN

- 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
- Batch Normalization & MaxPooling
- 3 Fully Connected layers with Dropout
- ~14M trainable parameters

## 📦 Project Structure

```
mlops_project/
├── mlops_cats_dogs_pipeline.ipynb  # Main notebook with all implementations
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── TEAM_SETUP.md                    # Team collaboration guide
├── DVC_REMOTE_SETUP.md             # DVC remote storage setup
├── .gitignore                       # Git ignore rules
├── dataset_metadata.json            # Dataset version info
├── data/
│   ├── raw/
│   │   └── PetImages.dvc           # DVC-tracked dataset (809 MB, 24,998 files)
│   └── processed/                   # Processed datasets
├── models/
│   ├── best_model.pt               # Best validation model
│   ├── baseline_cnn_final.pt       # Final trained model
│   ├── baseline_cnn_scripted.pt    # TorchScript model
│   ├── baseline_cnn.pkl            # Pickle format
│   └── baseline_cnn.onnx           # ONNX format (optional)
└── experiments/
    ├── mlruns/                      # MLflow tracking data
    ├── class_distribution.png       # EDA visualizations
    ├── sample_images.png
    ├── training_curves.png          # Loss & accuracy plots
    ├── confusion_matrix.png         # Model evaluation
    └── experiment_report.txt        # Summary report
```

## 🚀 Module M1: Model Development & Experiment Tracking

### Features Implemented

✅ **Data Versioning**: DVC for dataset tracking  
✅ **Data Pipeline**: Loading, cleaning, preprocessing (224x224 RGB)  
✅ **Data Augmentation**: Random flips, rotation, color jitter  
✅ **Model Training**: Baseline CNN with PyTorch  
✅ **Experiment Tracking**: MLflow for metrics, parameters, artifacts  
✅ **Evaluation**: Confusion matrix, classification report, loss curves  
✅ **Model Serialization**: .pt, .pkl, .onnx, TorchScript formats

### Technologies Used

- **ML Framework**: PyTorch
- **Experiment Tracking**: MLflow
- **Version Control**: Git (code) + DVC (data)
- **Data Processing**: NumPy, Pandas, PIL
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, recommended)
- Git
- DVC

### Installation Steps

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd mlops_project
```

2. **Get dataset with DVC** (Automatic - No manual setup needed!)

```bash
# If DVC remote is configured (see DVC_REMOTE_SETUP.md)
dvc pull  # Downloads 809 MB dataset automatically

# Otherwise, dataset is already in Git LFS or included in repository
```

**Note**: The dataset path is already configured in the notebook to use `data/raw/PetImages`. No manual path setup required!

3. **Create virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Install PyTorch** (choose based on your system)

```bash
# CPU only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### DVC Status

✅ **Dataset tracked**: `data/raw/PetImages` (809 MB, 24,998 images)  
✅ **DVC initialized**: Metadata in `data/raw/PetImages.dvc`  
❌ **Remote storage**: Not configured yet

### Setup DVC Remote (Optional - For Team Sharing)

See [DVC_REMOTE_SETUP.md](DVC_REMOTE_SETUP.md) for detailed instructions.

**Quick Setup (Google Drive - Recommended)**

```bash
# 1. Create Google Drive folder and get folder ID
# 2. Configure remote
dvc remote add -d storage gdrive://YOUR_FOLDER_ID
git add .dvc/config
git commit -m "Add DVC remote"

# 3. Push dataset
dvc push

# Team members can now pull
dvc pull
```

**Other options**: AWS S3, Azure Blob, SSH Server (see DVC_REMOTE_SETUP.md)

## 📓 Running the Notebook

### Option 1: Jupyter Notebook

```bash
jupyter notebook mlops_cats_dogs_pipeline.ipynb
```

### Option 2: JupyterLab

```bash
jupyter lab mlops_cats_dogs_pipeline.ipynb
```

### Option 3: VS Code

Open `mlops_cats_dogs_pipeline.ipynb` in VS Code with Jupyter extension

## 📊 View Experiments with MLflow

Start MLflow UI:

```bash
cd C:\Users\swath\dataset\mlops_project
mlflow ui --backend-store-uri file:///C:/Users/swath/dataset/mlops_project/experiments
```

Then open browser at: **http://localhost:5000**

### MLflow Tracks:

- Hyperparameters (learning rate, batch size, epochs, etc.)
- Metrics (train/val loss, train/val accuracy per epoch)
- Artifacts (model files, plots, confusion matrix)
- Model registry

## 🎯 Model Performance

| Metric         | Value                     |
| -------------- | ------------------------- |
| Test Accuracy  | ~85-90% (after 10 epochs) |
| Model Size     | ~55 MB (PyTorch .pt)      |
| Inference Time | ~50ms per image (CPU)     |
| Parameters     | ~14M trainable            |

## 🔄 Data Versioning with DVC

### Track Dataset

```bash
dvc add path/to/dataset
git add dataset.dvc .gitignore
git commit -m "Track dataset with DVC"
```

### Pull Dataset

```bash
dvc pull
```

### Push to Remote Storage (optional)

```bash
dvc remote add -d myremote s3://mybucket/path
dvc push
```

## 📈 Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Image Size    | 224x224           |
| Batch Size    | 32                |
| Epochs        | 10                |
| Learning Rate | 0.001             |
| Optimizer     | Adam              |
| Loss Function | CrossEntropyLoss  |
| LR Scheduler  | ReduceLROnPlateau |

## 🔬 Experiment Tracking

All experiments are logged to MLflow with:

- **Parameters**: Model config, hyperparameters
- **Metrics**: Loss, accuracy (per epoch)
- **Artifacts**:
  - Model checkpoints (.pt, .pkl, .onnx)
  - Training curves
  - Confusion matrix
  - Sample predictions
  - Experiment report

## 🚀 Next Steps: Completed Modules

### ✅ M1: Model Development & Experiment Tracking (COMPLETE)

- ✓ Git + DVC versioning
- ✓ Baseline CNN model training
- ✓ MLflow experiment tracking
- ✓ Model serialization (.pt, .pkl, .onnx)

### ✅ M2: Containerization & Packaging (COMPLETE)

- ✓ FastAPI REST API for inference
- ✓ Docker image with dependencies
- ✓ Health check and prediction endpoints
- ✓ Model serving on port 8000

### ✅ M3: CI/CD Pipeline (COMPLETE)

- ✓ GitHub Actions workflow
- ✓ Automated testing (pytest)
- ✓ Docker image build & publish
- ✓ Container registry integration

### ✅ M4: Continuous Deployment (COMPLETE)

- ✓ Docker Compose deployment
- ✓ Kubernetes manifests (deployment + service)
- ✓ Automated deployment on main branch
- ✓ Post-deployment smoke tests
- ✓ Pipeline fails on test failures

**See**: [M4_DEPLOYMENT_GUIDE.md](M4_DEPLOYMENT_GUIDE.md), [M4_QUICK_REFERENCE.md](M4_QUICK_REFERENCE.md)

### ✅ M5: Monitoring, Logs & Final Submission (COMPLETE)

- ✓ Request/response logging
- ✓ Metrics tracking (request count, latency)
- ✓ /metrics endpoint for monitoring
- ✓ Performance tracking script
- ✓ Post-deployment accuracy monitoring
- ✓ No sensitive data in logs

**See**: [M5_MONITORING_GUIDE.md](M5_MONITORING_GUIDE.md), [M5_QUICK_REFERENCE.md](M5_QUICK_REFERENCE.md)

### 🔮 Future Enhancements (Optional)

- Cloud deployment (Azure ML / AWS SageMaker)
- Auto-scaling & load balancing
- Advanced monitoring dashboards (Grafana)
- Automated retraining pipelines

## 📝 Model Formats

The trained model is saved in multiple formats:

1. **PyTorch (.pt)** - Full checkpoint with optimizer state
2. **TorchScript (.pt)** - Optimized for production deployment
3. **Pickle (.pkl)** - Python serialization format
4. **ONNX (.onnx)** - Cross-platform interoperability

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**MLOps Engineer**

- Project: Cats vs Dogs Classification Pipeline
- Use Case: Pet Adoption Platform

## 🙏 Acknowledgments

- **Dataset**: Kaggle Cats and Dogs Dataset
- **Framework**: PyTorch
- **Experiment Tracking**: MLflow
- **Version Control**: Git + DVC

## 📞 Support

For issues or questions, please open an issue in the repository.

---
