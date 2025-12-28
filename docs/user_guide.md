# KneeXpert User Guide

## Introduction

KneeXpert is an AI-powered system designed to assist healthcare professionals in diagnosing knee joint diseases through medical image analysis. This guide provides instructions for using the system.

## System Requirements

### For Training/Development
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### For Web Application
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Internet connection (for API access)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd KneeXpert
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- Place knee images in `data/raw/` directory
- Organize by class in subdirectories (e.g., `Normal/`, `Pathological/`)

## Training Models

### Basic Training
```bash
python main_train.py --model resnet50 --data_dir data/processed
```

### Advanced Options
```bash
python main_train.py \
    --model efficientnet_b0 \
    --data_dir data/processed \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001
```

### Available Models
- `resnet50`: ResNet-50 architecture
- `efficientnet_b0`: EfficientNet-B0 architecture
- `vit_base`: Vision Transformer (ViT-Base)

## Using the Web Application

### 1. Start Backend Server
```bash
cd app/backend
python api.py
```
Server will start at `http://localhost:8000`

### 2. Open Frontend
- Open `app/frontend/index.html` in a web browser
- Or serve using a local web server

### 3. Upload and Analyze
1. Click "Choose Image" or drag & drop an image
2. Select model from dropdown
3. Optionally enable Grad-CAM visualization
4. Click "Analyze Image"
5. View results including:
   - Predicted class
   - Confidence score
   - Class probabilities
   - Grad-CAM heatmap (if enabled)

## API Usage

### Predict Endpoint
```bash
curl -X POST "http://localhost:8000/predict?model_name=resnet50&include_gradcam=true" \
     -F "file=@knee_image.jpg"
```

### Response Format
```json
{
    "predicted_class": "Pathological",
    "predicted_class_idx": 1,
    "confidence": 0.95,
    "probabilities": {
        "Normal": 0.05,
        "Pathological": 0.95
    },
    "gradcam": "data:image/png;base64,..."
}
```

## Data Preprocessing

### Manual Preprocessing
```python
from preprocessing.utils import load_image, normalize_image
from preprocessing.augment import get_augmentation_transforms

# Load and normalize image
image = load_image("path/to/image.jpg")
normalized = normalize_image(image, method="min_max")

# Apply augmentation
transforms = get_augmentation_transforms(training=True)
augmented, _ = apply_augmentation(normalized, transforms)
```

## Model Evaluation

### Evaluate Trained Model
```python
from training.evaluate import evaluate_model
from torch.utils.data import DataLoader

metrics = evaluate_model(
    model,
    test_loader,
    device="cuda",
    class_names=["Normal", "Pathological"],
    save_dir="results/"
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller image size
   - Enable gradient checkpointing

2. **Model Not Found**
   - Ensure model is trained first
   - Check checkpoint path in API

3. **Poor Performance**
   - Increase training data
   - Adjust hyperparameters
   - Try different architectures

## Best Practices

1. **Data Quality**
   - Use high-quality, properly labeled images
   - Ensure balanced class distribution
   - Validate data before training

2. **Model Selection**
   - Start with ResNet-50 (good baseline)
   - Try EfficientNet for efficiency
   - Use ViT for potentially better accuracy

3. **Evaluation**
   - Always use held-out test set
   - Report multiple metrics
   - Include confidence intervals

4. **Clinical Use**
   - This is a research tool
   - Not for direct clinical diagnosis
   - Always consult medical professionals

## Support

For issues or questions:
- Check documentation in `docs/` directory
- Review code comments
- Contact project maintainers

---

*Last updated: [Date]*

