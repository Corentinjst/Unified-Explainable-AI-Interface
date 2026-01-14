"""
XAI service for coordinating explanation generation.
"""

import numpy as np
from PIL import Image
import logging
from typing import Tuple, Dict

from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .gradcam_explainer import GradCAMExplainer
from ..model_loader import ModelLoader
from ..audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class XAIService:
    """
    Service for generating XAI explanations.

    Coordinates preprocessing, model loading, and explanation generation
    for all supported XAI methods (LIME, SHAP, Grad-CAM).
    """

    def __init__(self):
        """Initialize XAI service with all explainers."""
        self.explainers = {
            "LIME": LIMEExplainer(num_samples=1000),
            "SHAP": SHAPExplainer(),
            "Grad-CAM": GradCAMExplainer()
        }
        self.model_loader = ModelLoader()
        self.audio_processor = AudioProcessor()
        logger.info("XAI service initialized with LIME, SHAP, and Grad-CAM")

    def explain_prediction(
        self,
        file_path: str,
        file_type: str,
        model_name: str,
        xai_method: str
    ) -> Tuple[str, Dict]:
        """
        Generate XAI explanation for a prediction.

        This is the main entry point for generating explanations. It handles:
        - Preprocessing (audio → spectrogram, image → resize)
        - Model loading and prediction
        - Explanation generation

        Args:
            file_path: Path to the file (audio or image)
            file_type: Either "audio" or "image"
            model_name: Name of the model to explain
            xai_method: XAI method to use ("LIME", "SHAP", or "Grad-CAM")

        Returns:
            Tuple of (visualization_base64, metadata_dict)

        Raises:
            ValueError: If invalid file_type, model, or XAI method
            Exception: If preprocessing, prediction, or explanation fails
        """
        try:
            logger.info(f"Generating {xai_method} explanation for {file_type} with {model_name}")

            # Validate XAI method
            if xai_method not in self.explainers:
                raise ValueError(
                    f"Invalid XAI method: {xai_method}. "
                    f"Available methods: {list(self.explainers.keys())}"
                )

            # 1. Preprocess input
            logger.debug("Preprocessing input...")
            if file_type == "audio":
                # Convert audio to spectrogram
                input_data = self.audio_processor.wav_to_spectrogram(file_path)
            elif file_type == "image":
                # Load and resize image
                img = Image.open(file_path).convert('RGB')
                img = img.resize((224, 224))
                input_data = np.array(img)
            else:
                raise ValueError(f"Invalid file_type: {file_type}. Must be 'audio' or 'image'")

            logger.debug(f"Input preprocessed: shape={input_data.shape}, dtype={input_data.dtype}")

            # 2. Get model
            logger.debug(f"Loading model {model_name}...")
            model = self.model_loader.get_model(model_name, file_type)

            # 3. Get prediction (to determine predicted class)
            logger.debug("Running prediction to determine class...")
            input_batch = np.expand_dims(input_data, axis=0)
            prediction = model.predict(input_batch, verbose=0)
            probability = float(prediction[0][0])
            predicted_class = int(probability > 0.5)
            logger.debug(f"Predicted class: {predicted_class} (probability: {probability:.4f})")

            # 4. Generate explanation
            logger.debug(f"Generating {xai_method} explanation...")
            explainer = self.explainers[xai_method]
            visualization, metadata = explainer.explain(
                input_data,
                model,
                predicted_class
            )

            # Add common metadata
            metadata["predicted_class_index"] = predicted_class
            metadata["predicted_probability"] = round(probability, 4)
            metadata["file_type"] = file_type
            metadata["model_name"] = model_name

            logger.info(f"{xai_method} explanation generated successfully")

            return visualization, metadata

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating XAI explanation: {str(e)}")
            raise Exception(f"Failed to generate {xai_method} explanation: {str(e)}")

    def get_available_methods(self) -> list:
        """
        Get list of available XAI methods.

        Returns:
            List of method names
        """
        return list(self.explainers.keys())

    def validate_method(self, method_name: str) -> bool:
        """
        Check if an XAI method is available.

        Args:
            method_name: Name of the method

        Returns:
            True if method is available, False otherwise
        """
        return method_name in self.explainers
