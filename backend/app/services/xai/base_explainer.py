from abc import ABC, abstractmethod
import numpy as np
import io
import base64
from PIL import Image
from typing import Tuple, Dict

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, input_data: np.ndarray, model, predicted_class: int) -> Tuple[str, Dict]:
        pass

    @staticmethod
    def array_to_base64(img_array: np.ndarray) -> str:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        if img_array.ndim == 2:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array, mode='RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def normalize_array(arr: np.ndarray, target_min: float = 0, target_max: float = 1) -> np.ndarray:
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min == 0:
            return np.zeros_like(arr)
        normalized = (arr - arr_min) / (arr_max - arr_min)
        normalized = normalized * (target_max - target_min) + target_min
        return normalized
