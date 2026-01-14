"""
Model loader service for loading and managing ML models.
"""

import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional
import logging

from ..utils.compatibility import AUDIO_MODELS, IMAGE_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton service for loading and managing ML models."""

    _instance = None
    _models: Dict[str, Dict[str, tf.keras.Model]] = {
        "audio": {},
        "image": {}
    }
    _base_path = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model loader with base path."""
        if not self._initialized:
            # Set base path to project root / models
            self._base_path = Path(__file__).parent.parent.parent.parent / "models"
            logger.info(f"Model loader initialized with base path: {self._base_path}")
            self._initialized = True

    def load_all_models(self):
        """
        Load all models defined in compatibility.py at startup.

        Logs:
            - Success for each loaded model
            - Warning for models that fail to load
            - Summary of loaded vs failed models
        """
        logger.info("="*60)
        logger.info("Starting model loading process...")
        logger.info("="*60)

        total_models = 0
        loaded_models = 0
        failed_models = []

        # Load audio models
        logger.info(f"\n[AUDIO MODELS] Loading {len(AUDIO_MODELS)} audio models...")
        for key, model_info in AUDIO_MODELS.items():
            total_models += 1
            model_name = model_info["name"]
            model_path = self._base_path / model_info["path"]

            try:
                if not model_path.exists():
                    logger.warning(f"  ✗ {model_name}: Model file not found at {model_path}")
                    failed_models.append(f"{model_name} (file not found)")
                    continue

                logger.info(f"  Loading {model_name} from {model_path}...")
                model = tf.keras.models.load_model(str(model_path), compile=False)
                self._models["audio"][key] = model
                loaded_models += 1
                logger.info(f"  ✓ {model_name}: Loaded successfully (accuracy: {model_info['accuracy']:.2%})")

            except Exception as e:
                logger.error(f"  ✗ {model_name}: Failed to load - {str(e)}")
                failed_models.append(f"{model_name} ({str(e)[:50]})")

        # Load image models
        logger.info(f"\n[IMAGE MODELS] Loading {len(IMAGE_MODELS)} image models...")
        for key, model_info in IMAGE_MODELS.items():
            total_models += 1
            model_name = model_info["name"]
            model_path = self._base_path / model_info["path"]

            try:
                if not model_path.exists():
                    logger.warning(f"  ✗ {model_name}: Model file not found at {model_path}")
                    failed_models.append(f"{model_name} (file not found)")
                    continue

                logger.info(f"  Loading {model_name} from {model_path}...")
                model = tf.keras.models.load_model(str(model_path), compile=False)
                self._models["image"][key] = model
                loaded_models += 1
                logger.info(f"  ✓ {model_name}: Loaded successfully (accuracy: {model_info['accuracy']:.2%})")

            except Exception as e:
                logger.error(f"  ✗ {model_name}: Failed to load - {str(e)}")
                failed_models.append(f"{model_name} ({str(e)[:50]})")

        # Summary
        logger.info("\n" + "="*60)
        logger.info(f"MODEL LOADING SUMMARY:")
        logger.info(f"  Total models: {total_models}")
        logger.info(f"  Successfully loaded: {loaded_models}")
        logger.info(f"  Failed to load: {len(failed_models)}")

        if failed_models:
            logger.warning(f"\nFailed models:")
            for failed in failed_models:
                logger.warning(f"  - {failed}")

        if loaded_models == 0:
            logger.error("\nCRITICAL: No models were loaded successfully!")
            logger.error("Please check model files and paths in compatibility.py")
        else:
            logger.info(f"\n✓ Model loading complete - {loaded_models}/{total_models} models ready")

        logger.info("="*60)

    def get_model(self, model_name: str, file_type: str) -> tf.keras.Model:
        """
        Retrieve loaded model by name and file type.

        Args:
            model_name: Name of the model (e.g., "VGG16")
            file_type: Either "audio" or "image"

        Returns:
            Loaded TensorFlow model

        Raises:
            ValueError: If model not found or not loaded
        """
        if file_type not in ["audio", "image"]:
            raise ValueError(f"Invalid file_type: {file_type}. Must be 'audio' or 'image'")

        # Try to find model by matching name
        for key, model in self._models[file_type].items():
            # Get the model info to compare names
            model_dict = AUDIO_MODELS if file_type == "audio" else IMAGE_MODELS
            if key in model_dict and model_dict[key]["name"] == model_name:
                return model

        # If not found, raise error
        available = [
            model_dict[key]["name"]
            for key in self._models[file_type].keys()
            if key in (AUDIO_MODELS if file_type == "audio" else IMAGE_MODELS)
        ]

        raise ValueError(
            f"Model '{model_name}' not found or not loaded for {file_type}. "
            f"Available models: {available}"
        )

    def is_model_loaded(self, model_name: str, file_type: str) -> bool:
        """
        Check if a model is loaded and available.

        Args:
            model_name: Name of the model
            file_type: Either "audio" or "image"

        Returns:
            True if model is loaded, False otherwise
        """
        try:
            self.get_model(model_name, file_type)
            return True
        except ValueError:
            return False

    def get_loaded_models(self, file_type: Optional[str] = None) -> Dict[str, list]:
        """
        Get list of all loaded models.

        Args:
            file_type: Optional filter by "audio" or "image". If None, returns both.

        Returns:
            Dictionary with loaded model names by type
        """
        result = {}

        if file_type is None or file_type == "audio":
            audio_names = [
                AUDIO_MODELS[key]["name"]
                for key in self._models["audio"].keys()
                if key in AUDIO_MODELS
            ]
            result["audio"] = audio_names

        if file_type is None or file_type == "image":
            image_names = [
                IMAGE_MODELS[key]["name"]
                for key in self._models["image"].keys()
                if key in IMAGE_MODELS
            ]
            result["image"] = image_names

        return result
