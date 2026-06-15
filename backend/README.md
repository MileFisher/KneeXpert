# KneeXpert Backbone API

X-ray KL grading (0–4) with ensemble learning and Grad-CAM; MRI with MACS-Net + DeiT-S.

## Layout

```
backbone/
├── app.py                 # FastAPI entry point
├── requirements.txt
├── shared/                # Cross-modality utilities
│   └── clinical_feedback.py
├── xray/                  # X-ray models & inference
│   ├── loader.py
│   ├── ensemble.py
│   ├── gradcam.py
│   └── models/            # *.pth weights (gitignored)
├── mri/                   # MRI pipeline
│   ├── pipeline.py
│   └── models/            # checkpoints (gitignored)
├── data/
│   ├── scan_label_summary.txt
│   └── samples/           # dev .nii.gz volumes (gitignored)
└── notebooks/
    └── mri_pipeline.ipynb
```

Place X-ray weights in `xray/models/` and MRI checkpoints in `mri/models/` (see filenames in `xray/loader.py` and `mri/config.py`).

## Setup

```bash
cd KneeXpert-Backbone
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 9000 --reload
```

Health check: `GET http://localhost:9000/health`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status and available models |
| POST | `/api/xray/predict` | Single image (KneeXpert) — `file`, optional `model_names` |
| POST | `/api/mri/predict` | Single MRI volume |
| POST | `/api/mri/predict/sample` | Dev sample from `data/samples/Effusion.nii.gz` |
| POST | `/predict` | Batch images (demo-compatible) |

**X-ray models** (weights in `xray/models/`):

| ID | Weights file |
|----|----------------|
| `densenet201_deep_mlp_v1` | `densenet201-deep_mlp_v1.pth` |
| `densenet201_deep_mlp_v2` | `densenet201-deep_mlp_v2.pth` |
| `densenet201_standard_head` | `densenet201-standard_heap.pth` |
| `resnet101_linear_head` | `resnet101-linear_head.pth` |
| `resnet50_deep_mlp` | `resnet50-deep_mlp.pth` |
| `resnet50_dropout_regularised` | `resnet50-dropout_regularised.pth` |
| `vgg19_batch_normalised` | `vgg19-batch_normalised.pth` |
| `vgg19_standard_head` | `vgg19-standard_head.pth` |

Pass `model_names=all` (default) to evaluate every available checkpoint. Ensemble = mean softmax across selected models.

**MRI models** (weights in `mri/models/`):

| ID | Weights file |
|----|----------------|
| `best_deit_small_multilabel_main` | `best_deit_small_multilabel_main.pth` |
| `best_macs_net`| `best_macs_net.pth` |