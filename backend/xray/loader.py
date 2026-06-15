"""
X-ray model loader — weights live under xray/models/.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

XRAY_ROOT = Path(__file__).resolve().parent
XRAY_WEIGHTS = XRAY_ROOT / "models"

CLASS_NAMES = ["0", "1", "2", "3", "4"]
NUM_CLASSES = len(CLASS_NAMES)

MODELS_CONFIG: dict[str, dict] = {
    "densenet201_deep_mlp_v1": {
        "path": XRAY_WEIGHTS / "densenet201-deep_mlp_v1.pth",
        "display_name": "DenseNet201 · Deep MLP v1",
        "family": "densenet201",
        "variant": "standard_linear",
        "is_custom": False,
        "type": "densenet201",
    },
    "densenet201_deep_mlp_v2": {
        "path": XRAY_WEIGHTS / "densenet201-deep_mlp_v2.pth",
        "display_name": "DenseNet201 · Deep MLP v2",
        "family": "densenet201",
        "variant": "deep_mlp",
        "is_custom": True,
        "custom_arch": "densenet201",
    },
    "densenet201_standard_head": {
        "path": XRAY_WEIGHTS / "densenet201-standard_heap.pth",
        "display_name": "DenseNet201 · Standard Head",
        "family": "densenet201",
        "variant": "deep_mlp",
        "is_custom": True,
        "custom_arch": "densenet201",
    },
    "resnet101_linear_head": {
        "path": XRAY_WEIGHTS / "resnet101-linear_head.pth",
        "display_name": "ResNet101 · Linear Head",
        "family": "resnet101",
        "variant": "linear",
        "is_custom": False,
        "type": "resnet101",
    },
    "resnet50_deep_mlp": {
        "path": XRAY_WEIGHTS / "resnet50-deep_mlp.pth",
        "display_name": "ResNet50 · Deep MLP",
        "family": "resnet50",
        "variant": "deep_mlp",
        "is_custom": True,
        "custom_arch": "resnet50",
    },
    "resnet50_dropout_regularised": {
        "path": XRAY_WEIGHTS / "resnet50-dropout_regularised.pth",
        "display_name": "ResNet50 · Dropout Regularised",
        "family": "resnet50",
        "variant": "deep_mlp",
        "is_custom": True,
        "custom_arch": "resnet50",
    },
    "vgg19_batch_normalised": {
        "path": XRAY_WEIGHTS / "vgg19-batch_normalised.pth",
        "display_name": "VGG19 · Batch Normalised",
        "family": "vgg19",
        "variant": "bn_head",
        "is_custom": False,
        "type": "vgg19_bn",
    },
    "vgg19_standard_head": {
        "path": XRAY_WEIGHTS / "vgg19-standard_head.pth",
        "display_name": "VGG19 · Standard Head",
        "family": "vgg19",
        "variant": "deep_mlp",
        "is_custom": True,
        "custom_arch": "vgg19",
    },
}

DEFAULT_ENSEMBLE_MODELS = [
    "densenet201_deep_mlp_v2",
    "resnet50_deep_mlp",
    "vgg19_batch_normalised",
]


def all_available_model_ids() -> list[str]:
    return [name for name, cfg in MODELS_CONFIG.items() if cfg["path"].exists()]


class ModelLoader:
    def __init__(self):
        self.models: dict[str, nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[xray] device: {self.device}")

    def _create_custom_model(self, model_type: str, num_classes: int = NUM_CLASSES):
        if model_type == "densenet201":
            backbone = models.densenet201(weights=None)
            num_ftrs = backbone.classifier.in_features
            backbone.classifier = nn.Identity()

            class CustomDenseNet(nn.Module):
                def __init__(self, backbone, num_ftrs, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(num_ftrs, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes),
                    )

                def forward(self, x):
                    x = self.backbone.features(x)
                    x = torch.nn.functional.relu(x, inplace=True)
                    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                    x = torch.flatten(x, 1)
                    return self.classifier(x)

            return CustomDenseNet(backbone, num_ftrs, num_classes)

        if model_type == "vgg19":

            class CustomVGG(nn.Module):
                def __init__(self, backbone, num_ftrs, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(num_ftrs, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes),
                    )

                def forward(self, x):
                    x = self.backbone.features(x)
                    x = self.backbone.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.backbone.classifier(x)
                    return self.classifier(x)

            backbone = models.vgg19(weights=None)
            num_ftrs = backbone.classifier[6].in_features
            backbone.classifier[6] = nn.Identity()
            return CustomVGG(backbone, num_ftrs, num_classes)

        if model_type == "resnet50":
            backbone = models.resnet50(weights=None)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
            return backbone

        return None

    def _create_standard_model(self, model_type: str, num_classes: int = NUM_CLASSES):
        if model_type == "resnet101":
            model = models.resnet101(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            return model
        if model_type == "densenet201":
            model = models.densenet201(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            return model
        if model_type == "vgg19_bn":
            model = models.vgg19_bn(weights=None)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
            return model
        return None

    def _load_model(self, model_name: str):
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model: {model_name}")
        config = MODELS_CONFIG[model_name]
        model_path = config["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if config["is_custom"]:
            model = self._create_custom_model(config["custom_arch"])
        else:
            model = self._create_standard_model(config["type"])

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        print(f"[xray] loading {model_name}...")
        model = self._load_model(model_name)
        self.models[model_name] = model
        print(f"[xray] loaded {model_name}")
        return model

    def get_config(self, model_name: str) -> dict:
        return MODELS_CONFIG[model_name]

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    def list_available(self) -> list[str]:
        return all_available_model_ids()

    def model_catalog(self) -> list[dict]:
        catalog = []
        for name, cfg in MODELS_CONFIG.items():
            if not cfg["path"].exists():
                continue
            catalog.append({
                "id": name,
                "display_name": cfg["display_name"],
                "family": cfg["family"],
                "variant": cfg["variant"],
                "weights_file": cfg["path"].name,
            })
        return catalog
