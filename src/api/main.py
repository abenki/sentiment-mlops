from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
from pathlib import Path
import time
from datetime import datetime
import logging

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.models.sentiment_model import SentimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'sentiment_predictions_total',
    'Total number of predictions made',
    ['sentiment']
)

ERROR_COUNTER = Counter(
    'sentiment_errors_total',
    'Total number of errors',
    ['error_type']
)

PREDICTION_LATENCY = Histogram(
    'sentiment_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

MODEL_LOAD_TIME = Gauge(
    'sentiment_model_load_time_seconds',
    'Time taken to load the model'
)

PREDICTION_CONFIDENCE = Histogram(
    'sentiment_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)


class PredictionRequest(BaseModel):
    """Prediction request"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! I loved every minute of it."
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    text: str
    sentiment: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    probabilities: dict = Field(..., description="Probabilities for each class")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This movie was great!",
                    "Terrible film, waste of time.",
                    "Not bad, could be better."
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
    timestamp: str


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = None
start_time = time.time()


def load_model():
    """Load the sentiment analysis model"""
    global model, model_version
    
    model_path = Path("models/sentiment_model.joblib")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info("Loading model...")
    load_start = time.time()
    
    try:
        model = joblib.load(model_path)
        load_time = time.time() - load_start
        MODEL_LOAD_TIME.set(load_time)
        
        model_version = "1.0.0"  # To improve with MLflow registry
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model version: {model_version}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Startup event for the application"""
    logger.info("Starting Sentiment Analysis API...")
    load_model()
    logger.info("API ready to serve predictions")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict sentiment for a single text
    
    - **text**: The text to analyze (1-10000 characters)
    
    Returns:
    - **sentiment**: 'positive' or 'negative'
    - **confidence**: Confidence score (0-1)
    - **probabilities**: Probability for each class
    """
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Prediction
        with PREDICTION_LATENCY.time():
            proba = model.predict_proba([request.text])[0]
            prediction = int(model.predict([request.text])[0])

        # Confidence calculation
        confidence = float(max(proba))
        
        # Sentiment
        sentiment = "positive" if prediction == 1 else "negative"
        
        # Metrics
        PREDICTION_COUNTER.labels(sentiment=sentiment).inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        
        # Processing time
        processing_time = (time.time() - start) * 1000
        
        return PredictionResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": float(proba[0]),
                "positive": float(proba[1])
            },
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts (max 100)
    
    - **texts**: List of texts to analyze
    
    Returns:
    - **predictions**: List of predictions for each text
    """
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    batch_start = time.time()
    predictions = []
    
    try:
        for text in request.texts:
            start = time.time()

            # Prediction
            proba = model.predict_proba([text])[0]
            prediction = int(model.predict([text])[0])
            
            confidence = float(max(proba))
            sentiment = "positive" if prediction == 1 else "negative"
            
            # Metrics
            PREDICTION_COUNTER.labels(sentiment=sentiment).inc()
            PREDICTION_CONFIDENCE.observe(confidence)
            
            processing_time = (time.time() - start) * 1000
            
            predictions.append(
                PredictionResponse(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    sentiment=sentiment,
                    confidence=confidence,
                    probabilities={
                        "negative": float(proba[0]),
                        "positive": float(proba[1])
                    },
                    processing_time_ms=processing_time,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        total_time = (time.time() - batch_start) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time
        )
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_version,
        "model_type": "Logistic Regression + TF-IDF",
        "vocabulary_size": len(model.vectorizer.get_feature_names_out()),
        "classes": ["negative", "positive"]
    }


@app.post("/debug/features", tags=["Debug"])
async def get_top_features(request: PredictionRequest, top_n: int = 10):
    """
    Get top N features (words) that influenced the prediction
    Useful for debugging and model interpretation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Vectorize the text
        X_vec = model.vectorizer.transform([request.text])

        # Get feature names and coefficients
        feature_names = model.vectorizer.get_feature_names_out()
        coefficients = model.classifier.coef_[0]

        # Get non-zero features for this text
        non_zero_indices = X_vec.nonzero()[1]

        # Calculate the impact of each feature
        feature_impacts = []
        for idx in non_zero_indices:
            feature_impacts.append({
                "word": feature_names[idx],
                "tfidf_score": float(X_vec[0, idx]),
                "coefficient": float(coefficients[idx]),
                "impact": float(X_vec[0, idx] * coefficients[idx])
            })

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        return {
            "text": request.text[:200],
            "top_features": feature_impacts[:top_n],
            "total_features_found": len(feature_impacts)
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
