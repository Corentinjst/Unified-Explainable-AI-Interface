import numpy as np
from PIL import Image
from typing import Dict

from .model_loader import ModelLoader
from .audio_processor import AudioProcessor

class InferenceService:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.audio_processor = AudioProcessor()

    def classify_image(self, image_path: str, model_name: str) -> Dict:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        model = self.model_loader.get_model(model_name, "image")
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
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
        return result

    def classify_audio(self, audio_path: str, model_name: str) -> Dict:
        spectrogram = self.audio_processor.wav_to_spectrogram(audio_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        model = self.model_loader.get_model(model_name, "audio")
        prediction = model.predict(spectrogram, verbose=0)
        probability = float(prediction[0][0])
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
        return result

    def get_preprocessed_input(self, file_path: str, file_type: str) -> np.ndarray:
        if file_type == "audio":
            return self.audio_processor.wav_to_spectrogram(file_path)
        elif file_type == "image":
            img = Image.open(file_path).convert('RGB')
            img = img.resize((224, 224))
            return np.array(img)
        else:
            raise ValueError(f"Invalid file_type: {file_type}")
