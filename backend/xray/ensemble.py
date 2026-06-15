"""
Ensemble inference: average softmax probabilities across selected models,
then pick argmax (majority-vote style). Grad-CAM per model + primary overlay from best-confidence model.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from shared.clinical_feedback import findings_for_grade
from xray.gradcam import GradCAMGenerator
from xray.loader import CLASS_NAMES, DEFAULT_ENSEMBLE_MODELS, MODELS_CONFIG, ModelLoader, all_available_model_ids

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model_loader: ModelLoader | None = None
_gradcam: GradCAMGenerator | None = None


def get_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def get_gradcam() -> GradCAMGenerator:
    global _gradcam
    if _gradcam is None:
        _gradcam = GradCAMGenerator()
    return _gradcam


def load_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def predict_xray(
    file_bytes: bytes,
    filename: str = "scan.jpg",
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run ensemble X-ray KL grading (0–4).
    Returns a dict ready for KneeXpert frontend consumption.
    """
    if model_names is None:
        model_names = list(all_available_model_ids())

    loader = get_loader()
    gradcam_gen = get_gradcam()
    image = load_image(file_bytes)
    input_cpu = TRANSFORM(image).unsqueeze(0)

    all_probs: list[np.ndarray] = []
    individual: dict[str, Any] = {}
    best_cam_model: str | None = None
    best_cam_conf = -1.0

    for m_name in model_names:
        model = loader.get_model(m_name)
        if model is None:
            continue

        device = next(model.parameters()).device
        input_tensor = input_cpu.to(device)
        input_tensor.requires_grad = True

        with torch.enable_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        probs_np = probabilities[0].detach().cpu().numpy()
        all_probs.append(probs_np)

        conf, pred_idx = torch.max(probabilities, 1)
        pred_class = CLASS_NAMES[pred_idx.item()]
        conf_score = float(conf.item() * 100)

        cfg = MODELS_CONFIG.get(m_name, {})
        family = cfg.get("family", m_name)
        gradcam_b64 = gradcam_gen.generate(model, family, input_tensor, image)

        individual[m_name] = {
            "model_id": m_name,
            "display_name": cfg.get("display_name", m_name),
            "family": family,
            "variant": cfg.get("variant", ""),
            "predicted_class": pred_class,
            "grade": int(pred_class),
            "confidence": round(conf_score, 2),
            "gradcam_base64": gradcam_b64,
            "class_probabilities": {
                CLASS_NAMES[i]: round(float(probs_np[i] * 100), 2) for i in range(len(CLASS_NAMES))
            },
        }

        if conf_score > best_cam_conf and gradcam_b64:
            best_cam_conf = conf_score
            best_cam_model = m_name

    if not all_probs:
        raise RuntimeError("No models could be loaded for prediction.")

    avg_probs = np.mean(all_probs, axis=0)
    ensemble_idx = int(np.argmax(avg_probs))
    ensemble_conf = float(avg_probs[ensemble_idx] * 100)
    ensemble_grade = int(CLASS_NAMES[ensemble_idx])

    primary_gradcam = None
    if best_cam_model and best_cam_model in individual:
        primary_gradcam = individual[best_cam_model].get("gradcam_base64")

    return {
        "filename": filename,
        "is_ensemble": len(model_names) > 1,
        "models_used": model_names,
        "model_count": len(individual),
        "ensemble_display_name": "Ensemble (probability average)",
        "predicted_class": str(ensemble_grade),
        "grade": ensemble_grade,
        "confidence": round(ensemble_conf, 2),
        "is_reliable": ensemble_conf >= 70.0,
        "findings": findings_for_grade(ensemble_grade),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(avg_probs[i] * 100), 2) for i in range(len(CLASS_NAMES))
        },
        "individual_results": individual,
        "gradcam_base64": primary_gradcam,
    }
