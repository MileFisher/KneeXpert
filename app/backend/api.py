"""
FastAPI backend for KneeXpert application.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import ResNetModel, EfficientNetModel, VisionTransformerModel, EnsembleModel
from explainability.grad_cam import generate_gradcam_heatmap, visualize_gradcam, get_target_layer

app = FastAPI(title="KneeXpert API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, model_path: str, num_classes: int = 5):
    """Load a trained model."""
    if model_name == "resnet50":
        model = ResNetModel(num_classes=num_classes, pretrained=False)
    elif model_name == "efficientnet_b0":
        model = EfficientNetModel(num_classes=num_classes, pretrained=False)
    elif model_name == "vit_base":
        model = VisionTransformerModel(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize and convert to RGB
    image = image.convert("RGB").resize((image_size, image_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats (standard for pre-trained models)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    # In production, load models from checkpoint paths
    # For now, models will be loaded on first request
    print("KneeXpert API started")
    print(f"Using device: {device}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KneeXpert API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(device)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "resnet50",
    include_gradcam: bool = False,
):
    """
    Predict knee disease from uploaded image.
    
    Args:
        file: Uploaded image file
        model_name: Model to use (resnet50, efficientnet_b0, vit_base)
        include_gradcam: Whether to include Grad-CAM visualization
        
    Returns:
        Prediction results with probabilities and optional Grad-CAM
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image).to(device)
        
        # Load model if not already loaded
        if model_name not in models:
            # In production, load from actual checkpoint path
            model_path = f"models/pretrained/{model_name}_best.pth"
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_name} not found. Please train the model first."
                )
            models[model_name] = load_model(model_name, model_path)
        
        model = models[model_name]
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = int(output.argmax(dim=1).item())
        
        # Prepare response
        class_names = ["KL Grade 0", "KL Grade 1", "KL Grade 2", "KL Grade 3", "KL Grade 4"]
        result = {
            "predicted_class": class_names[predicted_class],
            "predicted_class_idx": int(predicted_class),
            "probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            "confidence": float(probabilities[predicted_class]),
        }
        
        # Generate Grad-CAM if requested
        if include_gradcam:
            try:
                target_layer = get_target_layer(model, model_name)
                heatmap = generate_gradcam_heatmap(
                    model, input_tensor, target_layer, predicted_class, str(device)
                )
                
                # Convert heatmap to base64 for JSON response
                import base64
                from io import BytesIO
                
                # Create visualization
                img_array = np.array(image)
                overlaid = visualize_gradcam(img_array, heatmap, alpha=0.4)
                
                # Convert to base64
                buffer = BytesIO()
                Image.fromarray(overlaid).save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                result["gradcam"] = f"data:image/png;base64,{img_base64}"
            except Exception as e:
                result["gradcam_error"] = str(e)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    available_models = []
    model_dir = Path("models/pretrained")
    
    if model_dir.exists():
        for model_file in model_dir.glob("*_best.pth"):
            model_name = model_file.stem.replace("_best", "")
            available_models.append({
                "name": model_name,
                "path": str(model_file),
                "loaded": model_name in models,
            })
    
    return {"available_models": available_models, "loaded_models": list(models.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

