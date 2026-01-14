"""
FastAPI main application with CORS configuration and routers.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload, classify, xai
from .utils.compatibility import AUDIO_MODELS, IMAGE_MODELS, XAI_METHODS
from .models.schemas import ModelsResponse, ModelInfo
from .services.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This replaces the deprecated @app.on_event("startup") decorator.
    """
    # Startup: Load ML models
    logger.info("Application startup - initializing ML models...")
    model_loader = ModelLoader()
    model_loader.load_all_models()
    logger.info("Application startup complete")

    yield

    # Shutdown: Cleanup if needed
    logger.info("Application shutdown")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Unified Explainable AI Interface",
    description="API for audio deepfake detection and lung cancer detection with XAI techniques",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(classify.router)
app.include_router(xai.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Unified Explainable AI Interface API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "classify": "/api/classify",
            "xai_explain": "/api/xai/explain",
            "xai_compare": "/api/xai/compare",
            "models": "/api/models",
            "xai_methods": "/api/xai/methods"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Get available models
@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """
    Get list of available models for audio and image classification.

    Returns:
        ModelsResponse with lists of audio and image models
    """
    audio_models = [
        ModelInfo(**model)
        for model in AUDIO_MODELS.values()
    ]

    image_models = [
        ModelInfo(**model)
        for model in IMAGE_MODELS.values()
    ]

    return ModelsResponse(
        audio_models=audio_models,
        image_models=image_models
    )


# Get available XAI methods
@app.get("/api/xai/methods")
async def get_xai_methods():
    """
    Get list of available XAI methods.

    Returns:
        Dictionary with XAI methods and their metadata
    """
    return {"methods": list(XAI_METHODS.values())}


# Debug endpoint to check loaded models
@app.get("/api/debug/loaded-models")
async def debug_loaded_models():
    """
    Debug endpoint to check which models are loaded in memory.

    Returns:
        Dictionary with loaded model information
    """
    model_loader = ModelLoader()
    loaded = model_loader.get_loaded_models()
    return {
        "loaded_models": loaded,
        "total_audio": len(loaded.get("audio", [])),
        "total_image": len(loaded.get("image", []))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
