import time
from fastapi import APIRouter, HTTPException
from ..models.schemas import ClassifyRequest, ClassifyResponse, PredictionResult
from ..utils.file_handler import get_file_path
from ..services.inference_service import InferenceService
from ..utils.compatibility import (
    get_model_by_name,
    get_compatible_xai_methods
)
import os

inference_service = InferenceService()
router = APIRouter(prefix="/api", tags=["classify"])

@router.post("/classify", response_model=ClassifyResponse)
async def classify_file(request: ClassifyRequest):
    start_time = time.time()
    file_path = get_file_path(request.file_id)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        file_type = "audio"
    elif ext in [".jpg", ".jpeg", ".png"]:
        file_type = "image"
    else:
        raise HTTPException(status_code=400, detail="Unknown file type")

    model = get_model_by_name(request.model_name, file_type)
    if not model:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found")

    if file_type == "audio":
        prediction = inference_service.classify_audio(file_path, request.model_name)
    else:
        prediction = inference_service.classify_image(file_path, request.model_name)

    compatible_xai = get_compatible_xai_methods(file_type)
    processing_time = (time.time() - start_time) * 1000

    return ClassifyResponse(
        prediction=PredictionResult(
            class_name=prediction["class"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"]
        ),
        model_name=model["name"],
        file_type=file_type,
        processing_time_ms=round(processing_time, 2),
        compatible_xai_methods=compatible_xai
    )
