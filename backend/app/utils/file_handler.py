import os
import uuid
import time
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException
from .compatibility import (
    validate_file_extension,
    validate_mime_type,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def save_uploaded_file(file: UploadFile, file_id: str) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    return str(file_path)

def determine_file_type(filename: str, content_type: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in ALLOWED_EXTENSIONS["audio"]:
        if validate_mime_type(content_type, "audio"):
            return "audio"
    if ext in ALLOWED_EXTENSIONS["image"]:
        if validate_mime_type(content_type, "image"):
            return "image"
    raise HTTPException(status_code=400, detail=f"Invalid file type")

async def validate_uploaded_file(file: UploadFile) -> tuple[str, int]:
    file_type = determine_file_type(file.filename, file.content_type)
    content = await file.read()
    file_size = len(content)
    await file.seek(0)
    max_size = MAX_FILE_SIZE[file_type]
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large")
    return file_type, file_size

def get_file_path(file_id: str) -> str:
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.stem == file_id:
            return str(file_path)
    raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

def generate_file_id() -> str:
    return str(uuid.uuid4())

def cleanup_old_files(max_age_hours: int = 24):
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
