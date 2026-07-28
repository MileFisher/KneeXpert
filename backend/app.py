"""
KneeXpert backbone API — X-ray ensemble + Grad-CAM, MRI MACS-Net + DeiT-S.

Run from this directory:
  uvicorn app:app --host 0.0.0.0 --port 9000 --reload
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from mri.config import LABEL_SUMMARY_PATH, SAMPLE_MRI_PATH
from xray.ensemble import predict_xray
from mri.label_summary import parse_label_summary
from mri.pipeline import mri_models_available, predict_mri
from xray import DEFAULT_ENSEMBLE_MODELS, MODELS_CONFIG, ModelLoader, all_available_model_ids
from feedback_engine import generate_feedback
from model_profiler import get_all_flops, get_model_flops_for_response

app = FastAPI(title="KneeXpert Backbone API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_loader = ModelLoader()
VALID_MODELS = set(_loader.list_available())


@app.get("/health")
async def health_check():
    mri_status = mri_models_available()
    catalog = _loader.model_catalog()
    flops_data = get_all_flops()

    # Enrich model catalog with compute cost (both MAC and FLOP conventions)
    for entry in catalog:
        model_id = entry["id"]
        if model_id in flops_data.get("xray_models", {}):
            entry["gmacs"] = flops_data["xray_models"][model_id]["gmacs"]
            entry["gflops"] = flops_data["xray_models"][model_id]["gflops"]
            entry["params_m"] = flops_data["xray_models"][model_id]["params_m"]

    return {
        "status": "healthy",
        "service": "kneexpert-backbone",
        "xray_models_available": _loader.list_available(),
        "model_catalog": catalog,
        "default_ensemble": DEFAULT_ENSEMBLE_MODELS,
        "mri_models_available": mri_status,
        "mri_pipeline_ready": all(mri_status.values()),
        "mri_sample_available": SAMPLE_MRI_PATH.is_file(),
        "mri_sample_filename": SAMPLE_MRI_PATH.name if SAMPLE_MRI_PATH.is_file() else None,
        "mri_label_summary_available": LABEL_SUMMARY_PATH.is_file(),
        "model_flops": flops_data,
    }


@app.get("/api/mri/categories")
async def mri_categories():
    categories, _ = parse_label_summary()
    if not categories:
        raise HTTPException(status_code=404, detail="scan_label_summary.txt not found in backbone/data/")
    return {"categories": categories, "source": "data/scan_label_summary.txt"}


@app.post("/api/xray/predict")
async def predict_xray_single(
    file: UploadFile = File(...),
    model_names: str = Form(default="all"),
):
    """Single-image X-ray prediction for KneeXpert diagnostics workspace."""
    try:
        raw = model_names.strip().lower()
        if raw in ("", "all", "*"):
            selected = all_available_model_ids()
        else:
            selected = [m.strip() for m in model_names.split(",") if m.strip()]
        if not selected:
            selected = all_available_model_ids()
        for m in selected:
            if m not in MODELS_CONFIG:
                raise HTTPException(status_code=400, detail=f"Unknown model: {m}")
            if m not in _loader.list_available():
                raise HTTPException(status_code=503, detail=f"Model weights missing: {m}")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file.")

        result = predict_xray(contents, filename=file.filename or "scan.jpg", model_names=selected)

        # Inject structured feedback
        result["feedback"] = generate_feedback(
            grade=result.get("grade", 0),
            modality="xray",
            confidence=result.get("confidence", 0),
        )

        # Inject model FLOPs
        result["model_flops"] = get_model_flops_for_response(result.get("models_used", selected))

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e


@app.post("/api/mri/predict")
async def predict_mri_single(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
):
    """Single-volume MRI prediction: MACS-Net artifact removal → DeiT-S 2.5D classification."""
    try:
        mri_status = mri_models_available()
        if not all(mri_status.values()):
            missing = [k for k, ok in mri_status.items() if not ok]
            raise HTTPException(
                status_code=503,
                detail=f"MRI model weights missing: {', '.join(missing)}. Place checkpoints in backbone/mri/models/.",
            )

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file.")

        kwargs = {"threshold": threshold} if threshold is not None else {}
        result = predict_mri(file_bytes=contents, filename=file.filename or "scan.nii.gz", **kwargs)

        # Inject structured feedback from MRI predictions
        positive_names = [
            d["name"] for d in result.get("multilabel_predictions", []) if d.get("predicted")
        ]
        result["feedback"] = generate_feedback(
            grade=result.get("grade", 0),
            modality="mri",
            confidence=result.get("confidence", 0),
            positive_labels=positive_names,
        )

        # Inject model compute cost — slice count drives the study-level total
        result["model_flops"] = get_model_flops_for_response(
            result.get("models_used", []),
            sampled_slices=result.get("slices_processed"),
        )

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI prediction error: {e}") from e


@app.post("/api/mri/predict/sample")
async def predict_mri_sample(threshold: float | None = Form(default=None)):
    """Run MRI pipeline on the pre-loaded sample volume in the backbone folder (no upload)."""
    try:
        if not SAMPLE_MRI_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Sample MRI not found: {SAMPLE_MRI_PATH.name}. Place it in backbone/data/samples/.",
            )
        mri_status = mri_models_available()
        if not all(mri_status.values()):
            missing = [k for k, ok in mri_status.items() if not ok]
            raise HTTPException(status_code=503, detail=f"MRI model weights missing: {', '.join(missing)}")

        kwargs = {"threshold": threshold} if threshold is not None else {}
        result = predict_mri(file_path=str(SAMPLE_MRI_PATH), filename=SAMPLE_MRI_PATH.name, **kwargs)
        result["sample_mode"] = True

        # Inject structured feedback from MRI predictions
        positive_names = [
            d["name"] for d in result.get("multilabel_predictions", []) if d.get("predicted")
        ]
        result["feedback"] = generate_feedback(
            grade=result.get("grade", 0),
            modality="mri",
            confidence=result.get("confidence", 0),
            positive_labels=positive_names,
        )

        # Inject model compute cost — slice count drives the study-level total
        result["model_flops"] = get_model_flops_for_response(
            result.get("models_used", []),
            sampled_slices=result.get("slices_processed"),
        )

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI sample prediction error: {e}") from e


@app.post("/predict")
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_names: str = Form(...),
):
    """Batch predict (compatible with model-training demo UI)."""
    try:
        selected = [m.strip() for m in model_names.split(",") if m.strip()]
        if not selected:
            raise HTTPException(status_code=400, detail="No models selected.")

        batch_results = []
        for file in files:
            contents = await file.read()
            if not contents:
                continue
            try:
                r = predict_xray(contents, filename=file.filename or "scan.jpg", model_names=selected)
                r["feedback"] = generate_feedback(
                    grade=r.get("grade", 0),
                    modality="xray",
                    confidence=r.get("confidence", 0),
                )
                r["model_flops"] = get_model_flops_for_response(r.get("models_used", selected))
                batch_results.append(r)
            except Exception:
                continue

        if not batch_results:
            raise HTTPException(status_code=400, detail="No valid images processed.")
        return JSONResponse(content={"results": batch_results})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
