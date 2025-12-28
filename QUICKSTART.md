# KneeXpert Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- 8GB+ RAM

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd KneeXpert
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Prepare Your Data

Organize your knee images in the following structure:
```
data/raw/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Pathological/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Preprocess Data (Optional)

If you need to preprocess images:
```python
from preprocessing.utils import create_data_splits

# Create train/val/test splits
splits = create_data_splits(
    "data/raw",
    "data/splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 3. Train a Model

Train ResNet-50 model:
```bash
python main_train.py --model resnet50 --data_dir data/processed
```

Train EfficientNet:
```bash
python main_train.py --model efficientnet_b0 --data_dir data/processed
```

Train Vision Transformer:
```bash
python main_train.py --model vit_base --data_dir data/processed
```

### 4. Use the Web Application

1. **Start the backend server:**
   ```bash
   cd app/backend
   python api.py
   ```
   The API will be available at `http://localhost:8000`

2. **Open the frontend:**
   - Open `app/frontend/index.html` in your web browser
   - Or use a local server:
     ```bash
     # Python 3
     cd app/frontend
     python -m http.server 8080
     ```
     Then open `http://localhost:8080` in your browser

3. **Upload and analyze images:**
   - Select a model from the dropdown
   - Upload a knee image
   - Click "Analyze Image"
   - View results with probabilities and Grad-CAM visualization

## Example Usage

### Training with Custom Parameters

```bash
python main_train.py \
    --model resnet50 \
    --data_dir data/processed \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001
```

### Using the API Directly

```python
import requests

# Upload and predict
with open("knee_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        params={"model_name": "resnet50", "include_gradcam": True},
        files={"file": f}
    )

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Project Structure

```
KneeXpert/
├── data/              # Dataset directories
├── models/            # Model architectures
├── preprocessing/     # Data preprocessing
├── training/          # Training scripts
├── explainability/    # Grad-CAM and explainability
├── app/               # Web application
│   ├── backend/       # FastAPI backend
│   └── frontend/      # Web frontend
├── notebooks/         # Jupyter notebooks
└── docs/             # Documentation
```

## Next Steps

1. **Read the full documentation:**
   - `docs/user_guide.md` - Complete user guide
   - `docs/methodology.md` - Methodology details
   - `docs/literature_review.md` - Research background

2. **Explore the notebooks:**
   - `notebooks/exploration.ipynb` - Data exploration
   - Create your own notebooks for experimentation

3. **Customize for your needs:**
   - Modify `training/config.py` for different configurations
   - Add new model architectures in `models/`
   - Extend the API in `app/backend/api.py`

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size in `training/config.py`
- Use smaller image size
- Train on CPU (slower but works)

**Model Not Found:**
- Ensure you've trained a model first
- Check that checkpoint files exist in `models/pretrained/`

**Import Errors:**
- Make sure all dependencies are installed
- Check Python version (3.8+)
- Verify you're in the project root directory

## Getting Help

- Check the documentation in `docs/`
- Review code comments
- Check GitHub issues (if applicable)

## License

Academic Research Project

---

Happy coding! 🦴

