"""
XAI (Explainable AI) services for generating model explanations.

Includes implementations for:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Grad-CAM (Gradient-weighted Class Activation Mapping)
"""

from .base_explainer import BaseExplainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .gradcam_explainer import GradCAMExplainer
from .xai_service import XAIService

__all__ = [
    "BaseExplainer",
    "LIMEExplainer",
    "SHAPExplainer",
    "GradCAMExplainer",
    "XAIService"
]
