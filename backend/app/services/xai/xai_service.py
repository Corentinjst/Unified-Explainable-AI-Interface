import numpy as np
from PIL import Image
from typing import Tuple, Dict

from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .gradcam_explainer import GradCAMExplainer
from ..model_loader import ModelLoader
from ..audio_processor import AudioProcessor

class XAIService:
    def __init__(self):
        self.explainers = {
            "LIME": LIMEExplainer(num_samples=1000),
            "SHAP": SHAPExplainer(),
            "Grad-CAM": GradCAMExplainer()
        }
        self.model_loader = ModelLoader()
        self.audio_processor = AudioProcessor()

    def explain_prediction(
        self,
        file_path: str,
        file_type: str,
        model_name: str,
        xai_method: str
    ) -> Tuple[str, Dict]:
        if xai_method not in self.explainers:
            raise ValueError(f"Invalid XAI method: {xai_method}")

        if file_type == "audio":
            input_data = self.audio_processor.wav_to_spectrogram(file_path)
        elif file_type == "image":
            img = Image.open(file_path).convert('RGB')
            img = img.resize((224, 224))
            input_data = np.array(img)
        else:
            raise ValueError(f"Invalid file_type: {file_type}")

        model = self.model_loader.get_model(model_name, file_type)
        input_batch = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_batch, verbose=0)
        probability = float(prediction[0][0])
        predicted_class = int(probability > 0.5)

        explainer = self.explainers[xai_method]
        visualization, metadata = explainer.explain(
            input_data,
            model,
            predicted_class
        )

        metadata["predicted_class_index"] = predicted_class
        metadata["predicted_probability"] = round(probability, 4)
        metadata["file_type"] = file_type
        metadata["model_name"] = model_name

        return visualization, metadata

    def get_available_methods(self) -> list:
        return list(self.explainers.keys())

    def validate_method(self, method_name: str) -> bool:
        return method_name in self.explainers
