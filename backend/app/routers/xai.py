"""
XAI endpoint for generating explanations and comparisons (mock implementation).
"""

import time
import asyncio
from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    XAIRequest,
    XAIResponse,
    CompareRequest,
    CompareResponse,
    ComparisonResult
)
from ..utils.file_handler import get_file_path
from ..utils.mock_generator import generate_mock_xai_visualization, image_to_base64
from ..utils.compatibility import get_xai_method_by_name, XAI_METHODS
import os


router = APIRouter(prefix="/api/xai", tags=["xai"])


@router.post("/explain", response_model=XAIResponse)
async def explain_prediction(request: XAIRequest):
    """
    Generate XAI visualization for a classification.

    Args:
        request: XAIRequest with file_id, model_name, and xai_method

    Returns:
        XAIResponse with visualization and metadata
    """
    start_time = time.time()

    # Get file path
    file_path = get_file_path(request.file_id)

    # Validate XAI method
    xai_method = get_xai_method_by_name(request.xai_method)
    if not xai_method:
        raise HTTPException(
            status_code=400,
            detail=f"XAI method '{request.xai_method}' not found. Available: {list(XAI_METHODS.keys())}"
        )

    # For audio files, we would convert to spectrogram first
    # For now, assume we're working with images or spectrograms
    # Generate mock XAI visualization
    try:
        visualization, metadata = generate_mock_xai_visualization(
            file_path,
            request.xai_method
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating XAI visualization: {str(e)}"
        )

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms

    return XAIResponse(
        xai_method=request.xai_method,
        visualization=visualization,
        metadata=metadata,
        processing_time_ms=round(processing_time, 2)
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_xai_methods(request: CompareRequest):
    """
    Generate multiple XAI visualizations for comparison.

    Args:
        request: CompareRequest with file_id, model_name, and xai_methods list

    Returns:
        CompareResponse with multiple XAI visualizations
    """
    start_time = time.time()

    # Get file path
    file_path = get_file_path(request.file_id)

    # Validate all XAI methods
    for method_name in request.xai_methods:
        xai_method = get_xai_method_by_name(method_name)
        if not xai_method:
            raise HTTPException(
                status_code=400,
                detail=f"XAI method '{method_name}' not found"
            )

    # Generate visualizations for all methods
    comparisons = []
    for method_name in request.xai_methods:
        try:
            visualization, metadata = generate_mock_xai_visualization(
                file_path,
                method_name
            )
            comparisons.append(
                ComparisonResult(
                    xai_method=method_name,
                    visualization=visualization,
                    metadata=metadata
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating {method_name} visualization: {str(e)}"
            )

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms

    return CompareResponse(
        comparisons=comparisons,
        processing_time_ms=round(processing_time, 2)
    )


@router.get("/methods")
async def get_xai_methods():
    """
    Get list of available XAI methods.

    Returns:
        Dictionary of XAI methods with metadata
    """
    return {"methods": list(XAI_METHODS.values())}
