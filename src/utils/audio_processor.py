"""
Audio processing utilities for converting audio files to spectrograms
"""

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image


class AudioProcessor:
    """Processes audio files and converts them to spectrograms for model input"""
    
    def __init__(self, target_size=(224, 224), sample_rate=22050, n_mels=128):
        """
        Initialize audio processor
        
        Args:
            target_size: Target image size for spectrogram (height, width)
            sample_rate: Audio sample rate in Hz
            n_mels: Number of mel bands for mel-spectrogram
        """
        self.target_size = target_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
    
    def process(self, audio_path):
        """
        Process audio file and convert to spectrogram image
        
        Args:
            audio_path: Path to audio file (.wav)
            
        Returns:
            numpy array: Processed spectrogram as image (normalized 0-1)
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=5.0)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            fmax=8000
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create spectrogram image
        spectrogram_img = self._create_spectrogram_image(mel_spec_db)
        
        return spectrogram_img
    
    def _create_spectrogram_image(self, mel_spec_db):
        """
        Create an image from mel-spectrogram
        
        Args:
            mel_spec_db: Mel-spectrogram in dB scale
            
        Returns:
            numpy array: RGB image normalized to 0-1
        """
        # Create figure without axes
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Display spectrogram
        img = librosa.display.specshow(
            mel_spec_db,
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            fmax=8000,
            ax=ax,
            cmap='viridis'
        )
        
        # Remove axes for cleaner image
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        # Load as PIL Image and resize
        pil_img = Image.open(buf).convert('RGB')
        pil_img = pil_img.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(pil_img) / 255.0
        
        return img_array
    
    def create_display_spectrogram(self, audio_path):
        """
        Create a displayable spectrogram with axes and labels
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            PIL Image: Spectrogram with labels for display
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=5.0)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create figure with labels
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            fmax=8000,
            ax=ax,
            cmap='viridis'
        )
        
        ax.set_title('Mel-Spectrogram')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
