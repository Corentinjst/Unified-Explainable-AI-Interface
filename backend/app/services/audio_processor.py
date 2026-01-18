import librosa
import numpy as np
from PIL import Image
import io
import base64

class AudioProcessor:
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    TARGET_SIZE = (224, 224)

    @staticmethod
    def wav_to_spectrogram(audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=AudioProcessor.SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=AudioProcessor.N_FFT,
            hop_length=AudioProcessor.HOP_LENGTH,
            n_mels=AudioProcessor.N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (
            (mel_spec_db - mel_spec_db.min()) /
            (mel_spec_db.max() - mel_spec_db.min()) * 255
        )
        mel_spec_norm = mel_spec_norm.astype(np.uint8)
        img = Image.fromarray(mel_spec_norm)
        img = img.convert('RGB')
        img = img.resize(AudioProcessor.TARGET_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(img)
        return img_array

    @staticmethod
    def spectrogram_to_base64(spectrogram: np.ndarray) -> str:
        img = Image.fromarray(spectrogram.astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def validate_audio_file(audio_path: str) -> bool:
        try:
            y, sr = librosa.load(audio_path, sr=AudioProcessor.SAMPLE_RATE, duration=0.1)
            return len(y) > 0
        except:
            return False
