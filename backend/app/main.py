from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload, classify, xai
from .utils.compatibility import AUDIO_MODELS, IMAGE_MODELS, XAI_METHODS
from .models.schemas import ModelsResponse, ModelInfo
from .services.model_loader import ModelLoader

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader = ModelLoader()
    model_loader.load_all_models()
    yield

app = FastAPI(
    title="Unified Explainable AI Interface",
    description="API for audio deepfake detection and lung cancer detection with XAI techniques",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(classify.router)
app.include_router(xai.router)

@app.get("/")
async def root():
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    audio_models = [ModelInfo(**model) for model in AUDIO_MODELS.values()]
    image_models = [ModelInfo(**model) for model in IMAGE_MODELS.values()]
    return ModelsResponse(audio_models=audio_models, image_models=image_models)

@app.get("/api/xai/methods")
async def get_xai_methods():
    return {"methods": list(XAI_METHODS.values())}

@app.get("/api/debug/loaded-models")
async def debug_loaded_models():
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
