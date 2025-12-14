from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import joblib
import numpy as np
import mlflow
import mlflow.keras
from typing import List, Optional
import os

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(
    title="Breast Cancer Classification API",
    description="API for breast cancer prediction using MLP",
    version="1.0.0"
)

# ==========================
# GLOBAL VARIABLES
# ==========================
model = None
scaler = None
label_encoder = None

# ==========================
# MODELS
# ==========================
class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="30 features for breast cancer prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 
                           0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 
                           0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 
                           184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            }
        }


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="0 for benign, 1 for malignant")
    probability: float = Field(..., description="Probability of malignant")
    diagnosis: str = Field(..., description="Benign or Malignant")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool


# ==========================
# STARTUP EVENT
# ==========================
@app.on_event("startup")
async def load_models():
    """Load model and preprocessing artifacts on startup"""
    global model, scaler, label_encoder
    
    try:
        # Check which model to load (default to hypertuned)
        model_path = os.getenv("MODEL_PATH", "artifacts/models/hypertuned_mlp.keras")
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}, trying baseline...")
            model_path = "artifacts/models/baseline_mlp"
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Load scaler
        scaler_path = "artifacts/models/scaler.joblib"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠️  Scaler not found")
        
        # Load label encoder
        encoder_path = "artifacts/models/label_encoder.joblib"
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            print("✓ Label encoder loaded successfully")
        else:
            print("⚠️  Label encoder not found")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise


# ==========================
# ENDPOINTS
# ==========================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make a prediction on breast cancer data
    
    - **features**: List of 30 numerical features
    
    Returns:
    - **prediction**: 0 (benign) or 1 (malignant)
    - **probability**: Probability of being malignant
    - **diagnosis**: Human-readable diagnosis
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    
    try:
        # Validate input
        if len(input_data.features) != 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 30 features, got {len(input_data.features)}"
            )
        
        # Prepare input
        features = np.array(input_data.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        probability = float(model.predict(features_scaled, verbose=0)[0][0])
        prediction = int(probability > 0.5)
        
        # Get diagnosis label
        if label_encoder is not None:
            diagnosis = label_encoder.inverse_transform([prediction])[0]
        else:
            diagnosis = "Malignant" if prediction == 1 else "Benign"
        
        return {
            "prediction": prediction,
            "probability": probability,
            "diagnosis": diagnosis
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Multi-Layer Perceptron (MLP)",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params(),
        "layers": len(model.layers)
    }


# ==========================
# BATCH PREDICTION (Optional)
# ==========================
class BatchPredictionInput(BaseModel):
    samples: List[List[float]] = Field(..., description="Multiple samples for batch prediction")


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make predictions on multiple samples at once
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    
    try:
        results = []
        
        for sample in input_data.samples:
            if len(sample) != 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"Each sample must have 30 features, got {len(sample)}"
                )
            
            features = np.array(sample).reshape(1, -1)
            features_scaled = scaler.transform(features)
            probability = float(model.predict(features_scaled, verbose=0)[0][0])
            prediction = int(probability > 0.5)
            
            if label_encoder is not None:
                diagnosis = label_encoder.inverse_transform([prediction])[0]
            else:
                diagnosis = "Malignant" if prediction == 1 else "Benign"
            
            results.append({
                "prediction": prediction,
                "probability": probability,
                "diagnosis": diagnosis
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")