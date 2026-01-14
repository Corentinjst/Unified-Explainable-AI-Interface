"""
Audio processing service for converting WAV files to mel-spectrograms.

CRITICAL: This preprocessing MUST match exactly what was used during training
or predictions will be invalid.
"""

import librosa
import numpy as np
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio files into spectrograms for model input."""

    # Parameters MUST match training (from notebooks/00_audio_to_spectrogram.ipynb)
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    TARGET_SIZE = (224, 224)

    @staticmethod
    def wav_to_spectrogram(audio_path: str) -> np.ndarray:
        """
        Convert WAV audio to mel-spectrogram RGB image.

        This function replicates the exact preprocessing pipeline used during
        model training. Any deviation will produce invalid predictions.

        Process:
        1. Load audio with librosa at 22050 Hz
        2. Compute mel-spectrogram (n_fft=2048, hop_length=512, n_mels=128)
        3. Convert to log scale (dB)
        4. Normalize to [0, 255]
        5. Convert grayscale to RGB (duplicate to 3 channels)
        6. Resize to 224x224

        Args:
            audio_path: Path to WAV audio file

        Returns:
            numpy array of shape (224, 224, 3) with values in [0, 255]

        Raises:
            Exception: If audio file cannot be loaded or processed
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")

            # 1. Load audio with librosa
            y, sr = librosa.load(audio_path, sr=AudioProcessor.SAMPLE_RATE)
            logger.debug(f"Audio loaded: duration={len(y)/sr:.2f}s, sample_rate={sr}")

            # 2. Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=AudioProcessor.N_FFT,
                hop_length=AudioProcessor.HOP_LENGTH,
                n_mels=AudioProcessor.N_MELS
            )
            logger.debug(f"Mel-spectrogram computed: shape={mel_spec.shape}")

            # 3. Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            logger.debug(f"Converted to dB scale: range=[{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")

            # 4. Normalize to [0, 255]
            mel_spec_norm = (
                (mel_spec_db - mel_spec_db.min()) /
                (mel_spec_db.max() - mel_spec_db.min()) * 255
            )
            mel_spec_norm = mel_spec_norm.astype(np.uint8)
            logger.debug(f"Normalized: shape={mel_spec_norm.shape}, range=[{mel_spec_norm.min()}, {mel_spec_norm.max()}]")

            # 5. Convert to PIL Image and convert grayscale to RGB
            img = Image.fromarray(mel_spec_norm)
            img = img.convert('RGB')  # Grayscale to RGB (duplicates to 3 channels)
            logger.debug(f"Converted to RGB image: size={img.size}, mode={img.mode}")

            # 6. Resize to target size
            img = img.resize(AudioProcessor.TARGET_SIZE, Image.Resampling.BILINEAR)
            logger.debug(f"Resized to: {img.size}")

            # 7. Convert to numpy array
            img_array = np.array(img)
            logger.info(f"Spectrogram generated: shape={img_array.shape}, dtype={img_array.dtype}")

            return img_array

        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {str(e)}")
            raise Exception(f"Failed to process audio file: {str(e)}")

    @staticmethod
    def spectrogram_to_base64(spectrogram: np.ndarray) -> str:
        """
        Convert spectrogram array to base64 for visualization.

        This is useful for XAI methods that need to visualize the spectrogram
        alongside the explanation.

        Args:
            spectrogram: numpy array of shape (224, 224, 3)

        Returns:
            Base64-encoded PNG image with data URI prefix
        """
        try:
            # Convert numpy array to PIL Image
            img = Image.fromarray(spectrogram.astype(np.uint8))

            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error converting spectrogram to base64: {str(e)}")
            raise Exception(f"Failed to convert spectrogram: {str(e)}")

    @staticmethod
    def validate_audio_file(audio_path: str) -> bool:
        """
        Validate that an audio file can be loaded and processed.

        Args:
            audio_path: Path to audio file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to load the audio file
            y, sr = librosa.load(audio_path, sr=AudioProcessor.SAMPLE_RATE, duration=0.1)
            return len(y) > 0
        except Exception as e:
            logger.warning(f"Audio file validation failed: {str(e)}")
            return False
