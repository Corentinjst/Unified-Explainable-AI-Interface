"""
Classify endpoint for running model inference (mock implementation).
"""

import time
from fastapi import APIRouter, HTTPException
from ..models.schemas import ClassifyRequest, ClassifyResponse, PredictionResult
from ..utils.file_handler import get_file_path
from ..utils.mock_generator import generate_mock_classification
from ..utils.compatibility import (
    get_model_by_name,
    get_compatible_xai_methods
)
import os


router = APIRouter(prefix="/api", tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify_file(request: ClassifyRequest):
    """
    Run classification on uploaded file.

    Args:
        request: ClassifyRequest with file_id and model_name

    Returns:
        ClassifyResponse with prediction results and compatible XAI methods
    """
    start_time = time.time()

    # Get file path
    file_path = get_file_path(request.file_id)

    # Determine file type from extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        file_type = "audio"
    elif ext in [".jpg", ".jpeg", ".png"]:
        file_type = "image"
    else:
        raise HTTPException(status_code=400, detail="Unknown file type")

    # Validate model exists and is compatible
    model = get_model_by_name(request.model_name, file_type)
    if not model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not found or incompatible with {file_type}"
        )

    # Generate mock classification
    prediction = generate_mock_classification(file_type, request.model_name)

    # Get compatible XAI methods
    compatible_xai = get_compatible_xai_methods(file_type)

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms

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
