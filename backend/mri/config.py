"""MRI pipeline configuration — paths and inference hyperparameters."""

import os
from pathlib import Path
from dotenv import load_dotenv

MRI_ROOT = Path(__file__).resolve().parent
BACKBONE_ROOT = MRI_ROOT.parent

load_dotenv(BACKBONE_ROOT / ".env")

DATA_DIR = BACKBONE_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
WEIGHTS_DIR = MRI_ROOT / "models"
CATEGORIES_JSON = MRI_ROOT / "categories.json"
LABEL_SUMMARY_PATH = DATA_DIR / "scan_label_summary.txt"

MACS_CHECKPOINT = WEIGHTS_DIR / "best_macs_net.pth"
DEIT_CHECKPOINT = WEIGHTS_DIR / "best_deit_small_multilabel_main.pth"

MACS_IMG_SIZE = int(os.getenv("MRI_MACS_IMG_SIZE", "128"))
DEIT_IMG_SIZE = int(os.getenv("MRI_DEIT_IMG_SIZE", "224"))

SLICE_START_FRAC = float(os.getenv("MRI_SLICE_START_FRAC", "0.15"))
SLICE_END_FRAC = float(os.getenv("MRI_SLICE_END_FRAC", "0.85"))
MIN_FOREGROUND_RATIO = float(os.getenv("MRI_MIN_FOREGROUND_RATIO", "0.01"))
MAX_SAMPLES_PER_STUDY = int(os.getenv("MRI_MAX_SAMPLES_PER_STUDY", "12"))
MAX_GALLERY_SLICES = int(os.getenv("MRI_MAX_GALLERY_SLICES", "4"))

AGG_METHOD = os.getenv("MRI_AGG_METHOD", "max")
DROPOUT_RATE = float(os.getenv("MRI_DROPOUT_RATE", "0.25"))
DROP_PATH_RATE = float(os.getenv("MRI_DROP_PATH_RATE", "0.15"))

# DeiT was trained with MACS-cleaned input in the proposed (online) configuration.
PIPELINE_MODE = os.getenv("MRI_PIPELINE_MODE", "online")

# Pre-loaded dev sample (skip client upload during testing)
SAMPLE_MRI_FILENAME = os.getenv("MRI_SAMPLE_FILENAME", "Effusion.nii.gz")
SAMPLE_MRI_PATH = SAMPLES_DIR / SAMPLE_MRI_FILENAME
