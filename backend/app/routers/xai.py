"""
XAI endpoint for generating explanations and comparisons.
"""

import time
import logging
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

logger = logging.getLogger(__name__)

# Initialize XAI service
xai_service = XAIService()


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

    # Determine file type
    ext = os.path.splitext(file_path)[1].lower()
    file_type = "audio" if ext == ".wav" else "image"

    # Generate real XAI explanation
    try:
        logger.info(f"Generating {request.xai_method} explanation for {file_type}")
        visualization, metadata = xai_service.explain_prediction(
            file_path,
            file_type,
            request.model_name,
            request.xai_method
        )
        logger.info(f"{request.xai_method} explanation generated successfully")
    except ValueError as e:
        # Validation error (invalid model, method, etc.)
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # XAI generation error
        logger.error(f"XAI error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating XAI visualization: {str(e)}")

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

    # Determine file type
    ext = os.path.splitext(file_path)[1].lower()
    file_type = "audio" if ext == ".wav" else "image"

    # Generate visualizations for all methods
    comparisons = []
    for method_name in request.xai_methods:
        try:
            logger.info(f"Generating {method_name} explanation for comparison")
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
            logger.info(f"{method_name} explanation generated successfully")
        except ValueError as e:
            # Validation error
            logger.error(f"Validation error for {method_name}: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # XAI generation error
            logger.error(f"Error generating {method_name}: {str(e)}")
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
