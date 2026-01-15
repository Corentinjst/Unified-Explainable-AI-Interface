import time
from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    XAIRequest,
    XAIResponse,
    CompareRequest,
    CompareResponse,
    ComparisonResult
)
from ..utils.file_handler import get_file_path
from ..services.xai.xai_service import XAIService
from ..utils.compatibility import get_xai_method_by_name, XAI_METHODS
import os

xai_service = XAIService()
router = APIRouter(prefix="/api/xai", tags=["xai"])

@router.post("/explain", response_model=XAIResponse)
async def explain_prediction(request: XAIRequest):
    start_time = time.time()
    file_path = get_file_path(request.file_id)
    xai_method = get_xai_method_by_name(request.xai_method)
    if not xai_method:
        raise HTTPException(status_code=400, detail=f"XAI method '{request.xai_method}' not found")

    ext = os.path.splitext(file_path)[1].lower()
    file_type = "audio" if ext == ".wav" else "image"

    visualization, metadata = xai_service.explain_prediction(
        file_path,
        file_type,
        request.model_name,
        request.xai_method
    )

    processing_time = (time.time() - start_time) * 1000
    return XAIResponse(
        xai_method=request.xai_method,
        visualization=visualization,
        metadata=metadata,
        processing_time_ms=round(processing_time, 2)
    )

@router.post("/compare", response_model=CompareResponse)
async def compare_xai_methods(request: CompareRequest):
    start_time = time.time()
    file_path = get_file_path(request.file_id)

    for method_name in request.xai_methods:
        xai_method = get_xai_method_by_name(method_name)
        if not xai_method:
            raise HTTPException(status_code=400, detail=f"XAI method '{method_name}' not found")

    ext = os.path.splitext(file_path)[1].lower()
    file_type = "audio" if ext == ".wav" else "image"

    comparisons = []
    for method_name in request.xai_methods:
        visualization, metadata = xai_service.explain_prediction(
            file_path,
            file_type,
            request.model_name,
            method_name
        )
        comparisons.append(
            ComparisonResult(
                xai_method=method_name,
                visualization=visualization,
                metadata=metadata
            )
        )

    processing_time = (time.time() - start_time) * 1000
    return CompareResponse(
        comparisons=comparisons,
        processing_time_ms=round(processing_time, 2)
    )

@router.get("/methods")
async def get_xai_methods():
    return {"methods": list(XAI_METHODS.values())}
