import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional

from ..utils.compatibility import AUDIO_MODELS, IMAGE_MODELS

class ModelLoader:
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
        if not self._initialized:
            self._base_path = Path(__file__).parent.parent.parent.parent / "models"
            self._initialized = True

    def load_all_models(self):
        for key, model_info in AUDIO_MODELS.items():
            model_path = self._base_path / model_info["path"]
            try:
                if model_path.exists():
                    model = tf.keras.models.load_model(str(model_path), compile=False)
                    self._models["audio"][key] = model
            except:
                pass

        for key, model_info in IMAGE_MODELS.items():
            model_path = self._base_path / model_info["path"]
            try:
                if model_path.exists():
                    model = tf.keras.models.load_model(str(model_path), compile=False)
                    self._models["image"][key] = model
            except:
                pass

    def get_model(self, model_name: str, file_type: str) -> tf.keras.Model:
        for key, model in self._models[file_type].items():
            model_dict = AUDIO_MODELS if file_type == "audio" else IMAGE_MODELS
            if key in model_dict and model_dict[key]["name"] == model_name:
                return model
        raise ValueError(f"Model '{model_name}' not found")

    def is_model_loaded(self, model_name: str, file_type: str) -> bool:
        try:
            self.get_model(model_name, file_type)
            return True
        except ValueError:
            return False

    def get_loaded_models(self, file_type: Optional[str] = None) -> Dict[str, list]:
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
