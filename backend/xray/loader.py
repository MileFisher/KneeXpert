"""
X-ray model loader — weights live under xray/models/.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import models

XRAY_ROOT = Path(__file__).resolve().parent
BACKBONE_ROOT = XRAY_ROOT.parent
XRAY_WEIGHTS = XRAY_ROOT / "models"

load_dotenv(BACKBONE_ROOT / ".env")

CLASS_NAMES = ["0", "1", "2", "3", "4"]
NUM_CLASSES = len(CLASS_NAMES)

ARCH_CANONICAL = {
    "efficientnet": "EfficientNet",
    "densenet": "DenseNet",
    "resnet": "ResNet",
    "vgg": "VGG",
    "mobilenet": "MobileNet",
    "convnext": "ConvNeXt",
    "swin": "Swin",
    "vit": "ViT",
    "deit": "DeiT",
}


def _infer_arch(stem: str) -> tuple[str, str]:
    """Infer (family, arch_type) from a weight file stem."""
    s = stem.lower().replace("-", "_")
    if "swin_tiny" in s or "swin_t" in s:
        return "swin", "swin_tiny_patch4_window7_224"
    if "swin_small" in s or "swin_s" in s:
        return "swin", "swin_small_patch4_window7_224"
    if "swin_base" in s or "swin_b" in s:
        return "swin", "swin_base_patch4_window7_224"

    parts = s.split("_")
    try:
        import timm
        for i in range(len(parts), 0, -1):
            cand = "_".join(parts[:i])
            if timm.is_model(cand):
                return parts[0], cand
    except Exception:
        pass

    for fam in ("densenet201", "densenet121", "resnet101", "resnet50", "resnet18", "vgg19", "efficientnet_b3", "efficientnet_b0"):
        if fam in s:
            return fam, fam

    return parts[0], parts[0]


def _format_display_name(stem: str) -> str:
    s = stem.replace("-", "_")
    parts = s.split("_")
    first = parts[0].lower()
    for prefix, canon in ARCH_CANONICAL.items():
        if first.startswith(prefix):
            arch_prefix = canon
            suffix = first[len(prefix):]
            if suffix:
                arch_name = f"{arch_prefix}{suffix.title()}"
                rest_parts = parts[1:]
            elif len(parts) > 1:
                p1 = parts[1]
                p1_fmt = p1.upper() if len(p1) <= 2 else p1.title()
                arch_name = f"{arch_prefix}-{p1_fmt}"
                rest_parts = parts[2:]
            else:
                arch_name = arch_prefix
                rest_parts = []
            rest = " ".join(p.title() for p in rest_parts)
            return f"{arch_name} · {rest}" if rest else arch_name
    return stem.replace("-", " ").replace("_", " ").title()


# Custom architecture / head overrides for legacy checkpoints that cannot be
# inferred purely from the file stem.
LEGACY_OVERRIDES: dict[str, dict] = {
    "densenet201-deep_mlp_v1": {"is_custom": False, "type": "densenet201", "variant": "standard_linear"},
    "densenet201-deep_mlp_v2": {"is_custom": True, "custom_arch": "densenet201", "variant": "deep_mlp"},
    "densenet201-standard_heap": {"id": "densenet201_standard_head", "display_name": "DenseNet201 · Standard Head", "is_custom": True, "custom_arch": "densenet201", "variant": "deep_mlp"},
    "resnet101-linear_head": {"is_custom": False, "type": "resnet101", "variant": "linear"},
    "resnet50-deep_mlp": {"is_custom": True, "custom_arch": "resnet50", "variant": "deep_mlp"},
    "resnet50-dropout_regularised": {"is_custom": True, "custom_arch": "resnet50", "variant": "deep_mlp"},
    "vgg19-batch_normalised": {"is_custom": False, "type": "vgg19_bn", "variant": "bn_head"},
    "vgg19-standard_head": {"is_custom": True, "custom_arch": "vgg19", "variant": "deep_mlp"},
}
KNOWN_METADATA = LEGACY_OVERRIDES


def discover_models() -> dict[str, dict]:
    """Scan XRAY_WEIGHTS and discover all model files dynamically."""
    configs: dict[str, dict] = {}
    if not XRAY_WEIGHTS.exists():
        return configs

    for p in sorted(XRAY_WEIGHTS.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".pth", ".pt", ".safetensors", ".bin"):
            continue
        stem = p.stem
        meta = LEGACY_OVERRIDES.get(stem, {})
        model_id = meta.get("id", stem.replace("-", "_"))
        inferred_family, inferred_type = _infer_arch(stem)

        cfg = {
            "path": p,
            "display_name": meta.get("display_name", _format_display_name(stem)),
            "family": meta.get("family", inferred_family),
            "variant": meta.get(
                "variant",
                stem.split("-", 1)[1].replace("_", " ") if "-" in stem else stem.replace("_", " "),
            ),
            "is_custom": meta.get("is_custom", False),
        }
        if "type" in meta:
            cfg["type"] = meta["type"]
        elif not cfg["is_custom"]:
            cfg["type"] = inferred_type
        if "custom_arch" in meta:
            cfg["custom_arch"] = meta["custom_arch"]

        configs[model_id] = cfg
    return configs


MODELS_CONFIG: dict[str, dict] = discover_models()

_env_ensemble = os.getenv("DEFAULT_XRAY_ENSEMBLE")
if _env_ensemble:
    DEFAULT_ENSEMBLE_MODELS = [
        m.strip() for m in _env_ensemble.split(",") if m.strip() in MODELS_CONFIG
    ]
else:
    DEFAULT_ENSEMBLE_MODELS = [
        m for m in [
            "densenet201_deep_mlp_v2",
            "resnet50_deep_mlp",
            "vgg19_batch_normalised",
        ] if m in MODELS_CONFIG
    ] or list(MODELS_CONFIG.keys())[:3]


def all_available_model_ids() -> list[str]:
    MODELS_CONFIG.update(discover_models())
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
        # Try timm for modern architectures
        try:
            import timm
            if timm.is_model(model_type):
                return timm.create_model(model_type, num_classes=num_classes)
        except Exception:
            pass
        # Try torchvision (only callable model factory functions, not modules)
        if hasattr(models, model_type):
            model_fn = getattr(models, model_type)
            if callable(model_fn) and not isinstance(model_fn, type(models)):
                model = model_fn(weights=None)
                if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif hasattr(model, "classifier"):
                    if isinstance(model.classifier, nn.Linear):
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return model
        return None

    def _load_model(self, model_name: str):
        if model_name not in MODELS_CONFIG:
            MODELS_CONFIG.update(discover_models())
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model: {model_name}")
        config = MODELS_CONFIG[model_name]
        model_path = config["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if config.get("is_custom"):
            model = self._create_custom_model(config["custom_arch"])
        else:
            model = self._create_standard_model(config.get("type", config.get("family", "")))

        if model is None:
            raise ValueError(f"Could not construct model architecture for {model_name}")

        if str(model_path).endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        if model_name not in MODELS_CONFIG:
            MODELS_CONFIG.update(discover_models())
        print(f"[xray] loading {model_name}...")
        model = self._load_model(model_name)
        self.models[model_name] = model
        print(f"[xray] loaded {model_name}")
        return model

    def get_config(self, model_name: str) -> dict:
        if model_name not in MODELS_CONFIG:
            MODELS_CONFIG.update(discover_models())
        return MODELS_CONFIG[model_name]

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    def list_available(self) -> list[str]:
        return all_available_model_ids()

    def model_catalog(self) -> list[dict]:
        MODELS_CONFIG.update(discover_models())
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
