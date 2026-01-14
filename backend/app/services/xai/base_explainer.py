"""
Base explainer abstract class for XAI methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import io
import base64
from PIL import Image
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """
    Abstract base class for all XAI explainers.

    All explainers must implement the explain() method and can use
    the common utility methods for visualization.
    """

    @abstractmethod
    def explain(
        self,
        input_data: np.ndarray,
        model,
        predicted_class: int
    ) -> Tuple[str, Dict]:
        """
        Generate explanation for a prediction.

        Args:
            input_data: Preprocessed input (224, 224, 3) with values in [0, 255]
            model: Loaded TensorFlow model
            predicted_class: Predicted class index (0 or 1 for binary classification)

        Returns:
            Tuple of (visualization_base64, metadata_dict)
                - visualization_base64: Base64-encoded PNG image with data URI
                - metadata_dict: Method-specific metadata about the explanation
        """
        pass

    @staticmethod
    def array_to_base64(img_array: np.ndarray) -> str:
        """
        Convert numpy array to base64 image string.

        Args:
            img_array: numpy array of shape (H, W, 3) or (H, W) with values [0, 255]

        Returns:
            Base64-encoded PNG image with data URI prefix
        """
        try:
            # Ensure array is in valid range and type
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            # Handle grayscale vs RGB
            if img_array.ndim == 2:
                img = Image.fromarray(img_array, mode='L')
            else:
                img = Image.fromarray(img_array, mode='RGB')

            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error converting array to base64: {str(e)}")
            raise Exception(f"Failed to convert visualization: {str(e)}")

    @staticmethod
    def normalize_array(arr: np.ndarray, target_min: float = 0, target_max: float = 1) -> np.ndarray:
        """
        Normalize array to target range.

        Args:
            arr: Input array
            target_min: Minimum value of target range
            target_max: Maximum value of target range

        Returns:
            Normalized array
        """
        arr_min = arr.min()
        arr_max = arr.max()

        if arr_max - arr_min == 0:
            return np.zeros_like(arr)

        normalized = (arr - arr_min) / (arr_max - arr_min)
        normalized = normalized * (target_max - target_min) + target_min

        return normalized
