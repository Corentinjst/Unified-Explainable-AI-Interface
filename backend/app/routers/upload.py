import os
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from ..models.schemas import UploadResponse
from ..utils.file_handler import (
    validate_uploaded_file,
    save_uploaded_file,
    generate_file_id,
    get_file_path
)
from ..utils.compatibility import get_compatible_models
from ..services.audio_processor import AudioProcessor

router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    file_type, file_size = await validate_uploaded_file(file)
    file_id = generate_file_id()
    file_path = await save_uploaded_file(file, file_id)
    compatible_models = get_compatible_models(file_type)
    return UploadResponse(
        file_id=file_id,
        file_type=file_type,
        filename=file.filename,
        preview_url=f"/api/files/{file_id}",
        compatible_models=compatible_models
    )

@router.get("/files/{file_id}")
async def get_file(file_id: str, format: str = None):
    file_path = get_file_path(file_id)
    ext = os.path.splitext(file_path)[1].lower()
    if format == "spectrogram" and ext == ".wav":
        audio_processor = AudioProcessor()
        spectrogram = audio_processor.wav_to_spectrogram(file_path)
        img = Image.fromarray(spectrogram.astype('uint8'))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return Response(content=img_buffer.getvalue(), media_type="image/png")
    return FileResponse(file_path)
