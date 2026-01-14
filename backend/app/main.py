"""
FastAPI main application with CORS configuration and routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload, classify, xai
from .utils.compatibility import AUDIO_MODELS, IMAGE_MODELS, XAI_METHODS
from .models.schemas import ModelsResponse, ModelInfo


# Create FastAPI app
app = FastAPI(
    title="Unified Explainable AI Interface",
    description="API for audio deepfake detection and lung cancer detection with XAI techniques",
    version="1.0.0"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
