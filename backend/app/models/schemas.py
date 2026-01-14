"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# Upload endpoint models
class UploadResponse(BaseModel):
    file_id: str
    file_type: str  # 'audio' or 'image'
    filename: str
    preview_url: str
    compatible_models: List[Dict]


# Classify endpoint models
class ClassifyRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    file_id: str
    model_name: str


class PredictionResult(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]

    class Config:
        populate_by_name = True


class ClassifyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: PredictionResult
    model_name: str
    file_type: str
    processing_time_ms: float
    compatible_xai_methods: List[Dict]


# XAI endpoint models
class XAIRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    file_id: str
    model_name: str
    xai_method: str  # 'LIME', 'SHAP', or 'Grad-CAM'


class XAIResponse(BaseModel):
    xai_method: str
    visualization: str  # Base64-encoded image
    metadata: Dict
    processing_time_ms: float


# Compare endpoint models
class CompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    file_id: str
    model_name: str
    xai_methods: List[str]


class ComparisonResult(BaseModel):
    xai_method: str
    visualization: str
    metadata: Dict


class CompareResponse(BaseModel):
    comparisons: List[ComparisonResult]
    processing_time_ms: float


# Model info models
class ModelInfo(BaseModel):
    name: str
    type: str
    description: str
    input_format: str
    classes: List[str]
    accuracy: float


class XAIMethodInfo(BaseModel):
    name: str
    full_name: str
    compatible_with: List[str]
    description: str
    color_scheme: str


class ModelsResponse(BaseModel):
    audio_models: List[ModelInfo]
    image_models: List[ModelInfo]


class XAIMethodsResponse(BaseModel):
    methods: List[XAIMethodInfo]
