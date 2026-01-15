import numpy as np
from typing import Tuple, Dict
from lime import lime_image
from skimage.segmentation import mark_boundaries

from .base_explainer import BaseExplainer

class LIMEExplainer(BaseExplainer):
    def __init__(self, num_samples: int = 1000):
        self.explainer = lime_image.LimeImageExplainer()
        self.num_samples = num_samples

    def explain(self, input_data: np.ndarray, model, predicted_class: int) -> Tuple[str, Dict]:
        def predict_fn(images):
            predictions = model.predict(images, verbose=0)
            prob_class_1 = predictions.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])

        explanation = self.explainer.explain_instance(
            input_data.astype(np.float64),
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=self.num_samples
        )

        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )

        visualization = mark_boundaries(temp / 255.0, mask, color=(0, 1, 0), mode='thick')
        visualization = (visualization * 255).astype(np.uint8)

        feature_weights = explanation.local_exp[predicted_class]
        top_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)[:5]

        metadata = {
            "num_features": len(feature_weights),
            "num_superpixels": len(np.unique(mask)),
            "num_samples": self.num_samples,
            "top_features": [
                {"feature_id": int(feat), "weight": float(weight), "importance": "positive" if weight > 0 else "negative"}
                for feat, weight in top_features
            ]
        }

        visualization_base64 = self.array_to_base64(visualization)
        return visualization_base64, metadata
