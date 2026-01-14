"""
File handling utilities for upload validation and storage.
"""

import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException
from .compatibility import (
    validate_file_extension,
    validate_mime_type,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS
)


# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def save_uploaded_file(file: UploadFile, file_id: str) -> str:
    """
    Save uploaded file to disk.

    Args:
        file: FastAPI UploadFile object
        file_id: Unique identifier for the file

    Returns:
        Path to the saved file
    """
    # Get file extension
    ext = os.path.splitext(file.filename)[1].lower()

    # Create file path
    file_path = UPLOAD_DIR / f"{file_id}{ext}"

    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return str(file_path)


def determine_file_type(filename: str, content_type: str) -> str:
    """
    Determine if file is audio or image based on extension and MIME type.

    Args:
        filename: Name of the file
        content_type: MIME type of the file

    Returns:
        'audio' or 'image'

    Raises:
        HTTPException if file type cannot be determined
    """
    ext = os.path.splitext(filename)[1].lower()

    # Check audio
    if ext in ALLOWED_EXTENSIONS["audio"]:
        if validate_mime_type(content_type, "audio"):
            return "audio"

    # Check image
    if ext in ALLOWED_EXTENSIONS["image"]:
        if validate_mime_type(content_type, "image"):
            return "image"

    # If we get here, file type is invalid
    raise HTTPException(
        status_code=400,
        detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
    )


async def validate_uploaded_file(file: UploadFile) -> tuple[str, int]:
    """
    Validate uploaded file (type, size, etc.).

    Args:
        file: FastAPI UploadFile object

    Returns:
        Tuple of (file_type, file_size)

    Raises:
        HTTPException if validation fails
    """
    # Determine file type
    file_type = determine_file_type(file.filename, file.content_type)

    # Check file size
    # Read file content to check size
    content = await file.read()
    file_size = len(content)

    # Reset file pointer for later reading
    await file.seek(0)

    max_size = MAX_FILE_SIZE[file_type]
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size for {file_type}: {max_size / (1024*1024):.1f}MB"
        )

    return file_type, file_size


def get_file_path(file_id: str) -> str:
    """
    Get file path from file_id.

    Args:
        file_id: Unique identifier for the file

    Returns:
        Path to the file

    Raises:
        HTTPException if file not found
    """
    # Find file with matching file_id (any extension)
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.stem == file_id:
            return str(file_path)

    raise HTTPException(
        status_code=404,
        detail=f"File not found: {file_id}"
    )


def generate_file_id() -> str:
    """Generate unique file ID."""
    return str(uuid.uuid4())


def cleanup_old_files(max_age_hours: int = 24):
    """
    Clean up files older than specified hours.

    Args:
        max_age_hours: Maximum age in hours before deletion
    """
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
