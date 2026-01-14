"""
Upload endpoint for file uploads and validation.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..models.schemas import UploadResponse
from ..utils.file_handler import (
    validate_uploaded_file,
    save_uploaded_file,
    generate_file_id
)
from ..utils.compatibility import get_compatible_models


router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and validate audio (.wav) or image (.jpg, .png) file.

    Args:
        file: Uploaded file

    Returns:
        UploadResponse with file_id, file_type, and compatible models
    """
    # Validate file
    file_type, file_size = await validate_uploaded_file(file)

    # Generate unique file ID
    file_id = generate_file_id()

    # Save file
    file_path = await save_uploaded_file(file, file_id)

    # Get compatible models
    compatible_models = get_compatible_models(file_type)

    return UploadResponse(
        file_id=file_id,
        file_type=file_type,
        filename=file.filename,
        preview_url=f"/api/files/{file_id}",
        compatible_models=compatible_models
    )


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Retrieve uploaded file by ID.

    Args:
        file_id: Unique file identifier

    Returns:
        File content
    """
    from fastapi.responses import FileResponse
    from ..utils.file_handler import get_file_path

    file_path = get_file_path(file_id)
    return FileResponse(file_path)
