"""
Inference service for running model predictions on audio and image files.
"""

import numpy as np
from PIL import Image
import logging
from typing import Dict

from .model_loader import ModelLoader
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for running model inference on audio and image files."""

    def __init__(self):
        """Initialize inference service with model loader and audio processor."""
        self.model_loader = ModelLoader()
        self.audio_processor = AudioProcessor()

    def classify_image(self, image_path: str, model_name: str) -> Dict:
        """
        Run classification on image file.

        Process:
        1. Load and preprocess image (resize to 224x224)
        2. Get model from loader
        3. Run inference (model has built-in Rescaling layer)
        4. Format results

        Args:
            image_path: Path to image file (JPG, PNG)
            model_name: Name of model to use (e.g., "VGG16", "DenseNet121")

        Returns:
            Dictionary with:
                - class: str (predicted class name)
                - confidence: float (confidence in prediction, 0-1)
                - probabilities: dict (probability for each class)

        Raises:
            ValueError: If model not found or not loaded
            Exception: If image processing or inference fails
        """
        try:
            logger.info(f"Classifying image with {model_name}: {image_path}")

            # 1. Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)  # Shape: (224, 224, 3), values [0, 255]
            logger.debug(f"Image loaded: shape={img_array.shape}, dtype={img_array.dtype}")

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

            # 2. Get model (has Rescaling layer built-in, so feed [0, 255] values)
            model = self.model_loader.get_model(model_name, "image")
            logger.debug(f"Model loaded: {model_name}")

            # 3. Run inference
            prediction = model.predict(img_array, verbose=0)
            probability = float(prediction[0][0])
            logger.debug(f"Raw prediction: {probability:.4f}")

            # 4. Format results (binary classification)
            # Output: sigmoid probability in [0, 1]
            # 0 = benign, 1 = malignant
            classes = ["benign", "malignant"]
            predicted_class = classes[1] if probability > 0.5 else classes[0]
            confidence = probability if probability > 0.5 else 1 - probability

            result = {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    classes[0]: round(1 - probability, 4),
                    classes[1]: round(probability, 4)
                }
            }

            logger.info(f"Classification result: {predicted_class} (confidence: {confidence:.2%})")
            return result

        except ValueError as e:
            logger.error(f"Model error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Inference error for image: {str(e)}")
            raise Exception(f"Failed to classify image: {str(e)}")

    def classify_audio(self, audio_path: str, model_name: str) -> Dict:
        """
        Run classification on audio file.

        Process:
        1. Convert WAV to mel-spectrogram (matches training preprocessing)
        2. Get model from loader
        3. Run inference on spectrogram
        4. Format results

        Args:
            audio_path: Path to WAV audio file
            model_name: Name of model to use (e.g., "VGG16", "ResNet50")

        Returns:
            Dictionary with:
                - class: str (predicted class name)
                - confidence: float (confidence in prediction, 0-1)
                - probabilities: dict (probability for each class)

        Raises:
            ValueError: If model not found or not loaded
            Exception: If audio processing or inference fails
        """
        try:
            logger.info(f"Classifying audio with {model_name}: {audio_path}")

            # 1. Convert audio to spectrogram
            # CRITICAL: This must match training preprocessing exactly
            spectrogram = self.audio_processor.wav_to_spectrogram(audio_path)
            logger.debug(f"Spectrogram generated: shape={spectrogram.shape}, dtype={spectrogram.dtype}")

            # Add batch dimension
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Shape: (1, 224, 224, 3)

            # 2. Get model
            model = self.model_loader.get_model(model_name, "audio")
            logger.debug(f"Model loaded: {model_name}")

            # 3. Run inference
            prediction = model.predict(spectrogram, verbose=0)
            probability = float(prediction[0][0])
            logger.debug(f"Raw prediction: {probability:.4f}")

            # 4. Format results (binary classification)
            # Output: sigmoid probability in [0, 1]
            # 0 = fake, 1 = real
            classes = ["fake", "real"]
            predicted_class = classes[1] if probability > 0.5 else classes[0]
            confidence = probability if probability > 0.5 else 1 - probability

            result = {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    classes[0]: round(1 - probability, 4),
                    classes[1]: round(probability, 4)
                }
            }

            logger.info(f"Classification result: {predicted_class} (confidence: {confidence:.2%})")
            return result

        except ValueError as e:
            logger.error(f"Model error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Inference error for audio: {str(e)}")
            raise Exception(f"Failed to classify audio: {str(e)}")

    def get_preprocessed_input(self, file_path: str, file_type: str) -> np.ndarray:
        """
        Get preprocessed input for XAI methods.

        This returns the exact input that will be fed to the model,
        which is needed for XAI explanation generation.

        Args:
            file_path: Path to file
            file_type: Either "audio" or "image"

        Returns:
            numpy array of shape (224, 224, 3) with values in [0, 255]

        Raises:
            ValueError: If file_type is invalid
            Exception: If preprocessing fails
        """
        try:
            if file_type == "audio":
                # Convert audio to spectrogram
                return self.audio_processor.wav_to_spectrogram(file_path)

            elif file_type == "image":
                # Load and resize image
                img = Image.open(file_path).convert('RGB')
                img = img.resize((224, 224))
                return np.array(img)

            else:
                raise ValueError(f"Invalid file_type: {file_type}. Must be 'audio' or 'image'")

        except Exception as e:
            logger.error(f"Error preprocessing {file_type} file: {str(e)}")
            raise Exception(f"Failed to preprocess {file_type}: {str(e)}")
