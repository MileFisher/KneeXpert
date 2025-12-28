# Sharing the Trained Model

The trained model files are located in `models/pretrained/resnet50/`:
- `best_model.pth` - Best model based on validation accuracy (46.13%)
- `final_model.pth` - Final model after training
- `training_history.json` - Training metrics and history
- `evaluation_metrics.json` - Test set evaluation results
- `confusion_matrix.png` - Confusion matrix visualization

## Model Details

- **Architecture**: ResNet-50
- **Classes**: 5 KL grades (0, 1, 2, 3, 4)
- **Input Size**: 224x224 RGB images
- **Normalization**: ImageNet mean/std
- **Validation Accuracy**: 46.13%
- **Test Accuracy**: 47.28%

## How to Share

### Option 1: GitHub Releases (Recommended)

1. Go to your GitHub repository
2. Click **"Releases"** → **"Create a new release"**
3. Tag: `v1.0.0`
4. Title: `ResNet50 Trained Model`
5. Upload `best_model.pth` as a release asset
6. Add description with model details

### Option 2: Git LFS

```powershell
# Install Git LFS: https://git-lfs.github.com/
git lfs install
git lfs track "*.pth"
git add models/pretrained/resnet50/best_model.pth
git add .gitattributes
git commit -m "Add trained ResNet50 model"
git push
```

### Option 3: External Storage

Upload to:
- Google Drive
- Dropbox
- OneDrive
- AWS S3
- Azure Blob Storage

Then share the link in README.md

## How Others Can Use the Model

```python
import torch
from models import ResNetModel

# Load the model
class_names = ['0', '1', '2', '3', '4']
model = ResNetModel(
    num_classes=len(class_names),
    model_name="resnet50",
    pretrained=False,
    dropout=0.5,
    freeze_backbone=False,
)

# Load weights
checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use for inference
# (with proper image preprocessing: resize to 224x224, normalize)
```

## File Sizes

- `best_model.pth`: ~97 MB
- `final_model.pth`: ~97 MB
- Other files: < 1 MB each

Total: ~195 MB (too large for regular git, use LFS or releases)

