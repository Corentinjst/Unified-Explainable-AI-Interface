import numpy as np
from typing import Tuple, Dict
import shap
from skimage.segmentation import slic
from matplotlib import cm

from .base_explainer import BaseExplainer

class SHAPExplainer(BaseExplainer):
    def __init__(self):
        pass

    def explain(self, input_data: np.ndarray, model, predicted_class: int) -> Tuple[str, Dict]:
        n_segments = 50
        segments = slic(
            input_data.astype(np.float64) / 255.0,
            n_segments=n_segments,
            compactness=10,
            sigma=1,
            start_label=0
        )
        num_superpixels = int(segments.max() + 1)

        def mask_image(masks, original_image, segments, background=None):
            if background is None:
                background = np.full_like(original_image, 128)
            output = []
            for mask in masks:
                masked_img = background.copy()
                for i, on in enumerate(mask):
                    if on:
                        masked_img[segments == i] = original_image[segments == i]
                output.append(masked_img)
            return np.array(output)

        def predict_fn(masks):
            masked_images = mask_image(masks, input_data, segments)
            return model.predict(masked_images, verbose=0)

        background = np.zeros((1, num_superpixels))
        explainer = shap.KernelExplainer(predict_fn, background)
        input_mask = np.ones((1, num_superpixels))
        shap_values = explainer.shap_values(input_mask, nsamples=500)

        if isinstance(shap_values, list):
            superpixel_shap = shap_values[0][0]
        else:
            superpixel_shap = shap_values[0]

        if predicted_class == 0:
            superpixel_shap = -superpixel_shap

        shap_vals_aggregated = np.zeros((224, 224))
        for i in range(num_superpixels):
            shap_vals_aggregated[segments == i] = superpixel_shap[i]

        vmax = np.abs(shap_vals_aggregated).max()
        if vmax == 0:
            vmax = 1e-8
        vmin = -vmax

        shap_normalized = (shap_vals_aggregated - vmin) / (vmax - vmin)
        cmap = cm.get_cmap('RdBu_r')
        heatmap_colored = cmap(shap_normalized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        blended = (input_data * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

        metadata = {
            "mean_abs_shap": float(np.abs(shap_vals_aggregated).mean()),
            "max_shap": float(shap_vals_aggregated.max()),
            "min_shap": float(shap_vals_aggregated.min()),
            "positive_contribution_ratio": float((shap_vals_aggregated > 0).sum() / shap_vals_aggregated.size),
            "explainer_type": "KernelExplainer",
            "num_superpixels": num_superpixels,
            "nsamples": 500
        }

        visualization_base64 = self.array_to_base64(blended)
        return visualization_base64, metadata
