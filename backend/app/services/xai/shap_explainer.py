"""
SHAP (SHapley Additive exPlanations) explainer for deep learning models.
"""

import numpy as np
import logging
from typing import Tuple, Dict
import shap
from skimage.segmentation import slic
from matplotlib import cm

from .base_explainer import BaseExplainer

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP explainer for image classification models.

    SHAP uses game theory to explain predictions by computing the contribution
    of each feature using Shapley values. This implementation uses superpixel
    segmentation to reduce dimensionality, making KernelExplainer feasible
    for image data. It shows which regions of the image increase or decrease
    the prediction probability.
    """

    def __init__(self):
        """Initialize SHAP explainer."""
        logger.info("SHAP explainer initialized")

    def explain(
        self,
        input_data: np.ndarray,
        model,
        predicted_class: int
    ) -> Tuple[str, Dict]:
        """
        Generate SHAP explanation using KernelExplainer with superpixels.

        Process:
        1. Segment image into superpixels using SLIC
        2. Create mask function that maps superpixel on/off to images
        3. Initialize KernelExplainer with model wrapper
        4. Compute SHAP values for each superpixel
        5. Map SHAP values back to pixel space
        6. Create red/blue heatmap visualization
        7. Blend with original image
        8. Return as base64 + metadata

        Args:
            input_data: Image array (224, 224, 3) with values [0, 255]
            model: TensorFlow model
            predicted_class: Predicted class index (0 or 1)

        Returns:
            Tuple of (visualization_base64, metadata)
        """
        try:
            logger.info("Generating SHAP explanation...")

            # Step 1: Segment image into superpixels
            # Use SLIC algorithm for compact, uniform superpixels
            n_segments = 100  # Number of superpixels
            segments = slic(
                input_data.astype(np.float64) / 255.0,
                n_segments=n_segments,
                compactness=10,
                sigma=1,
                start_label=0
            )
            num_superpixels = segments.max() + 1
            logger.debug(f"Created {num_superpixels} superpixels")

            # Step 2: Create mask function
            # Maps binary mask (which superpixels are "on") to actual images
            def mask_image(masks, original_image, segments, background=None):
                """
                Apply masks to create images with some superpixels replaced.

                Args:
                    masks: Binary array (n_samples, num_superpixels)
                    original_image: Original image (224, 224, 3)
                    segments: Segmentation map (224, 224)
                    background: Background color (default: gray)

                Returns:
                    Masked images (n_samples, 224, 224, 3)
                """
                if background is None:
                    background = np.full_like(original_image, 128)  # Gray background

                output = []
                for mask in masks:
                    masked_img = background.copy()
                    for i, on in enumerate(mask):
                        if on:
                            masked_img[segments == i] = original_image[segments == i]
                    output.append(masked_img)
                return np.array(output)

            # Step 3: Create prediction function wrapper
            def predict_fn(masks):
                """Predict on masked images."""
                masked_images = mask_image(masks, input_data, segments)
                return model.predict(masked_images, verbose=0)

            # Step 4: Create background (all superpixels off = gray image)
            # For KernelExplainer, we use a single background sample
            background = np.zeros((1, num_superpixels))
            logger.debug("Created background mask (all superpixels off)")

            # Step 5: Initialize KernelExplainer
            logger.debug("Initializing SHAP KernelExplainer...")
            explainer = shap.KernelExplainer(predict_fn, background)

            # Step 6: Compute SHAP values for input (all superpixels on)
            input_mask = np.ones((1, num_superpixels))
            logger.debug("Computing SHAP values...")
            shap_values = explainer.shap_values(input_mask, nsamples=500)
            logger.debug("SHAP values computed")

            # Handle SHAP output format
            if isinstance(shap_values, list):
                # List of arrays, one per output class
                superpixel_shap = shap_values[predicted_class][0]  # (num_superpixels,)
            else:
                superpixel_shap = shap_values[0]  # (num_superpixels,)

            logger.debug(f"Superpixel SHAP values: {superpixel_shap.shape}, "
                        f"range: [{superpixel_shap.min():.4f}, {superpixel_shap.max():.4f}]")

            # Step 7: Map SHAP values back to pixel space
            shap_vals_aggregated = np.zeros((224, 224))
            for i in range(num_superpixels):
                shap_vals_aggregated[segments == i] = superpixel_shap[i]

            logger.debug("Mapped SHAP values to pixel space")

            # Step 8: Normalize for visualization
            # Center around 0, with symmetric range
            vmax = np.abs(shap_vals_aggregated).max()
            if vmax == 0:
                vmax = 1e-8  # Avoid division by zero
            vmin = -vmax

            # Normalize to [0, 1] for colormap
            shap_normalized = (shap_vals_aggregated - vmin) / (vmax - vmin)

            # Apply red/blue colormap
            # RdBu_r: Red for positive contribution, Blue for negative
            cmap = cm.get_cmap('RdBu_r')
            heatmap_colored = cmap(shap_normalized)[:, :, :3]  # RGB only (no alpha)
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            logger.debug("Applied RdBu_r colormap to SHAP values")

            # Blend with original image (50% opacity each)
            blended = (input_data * 0.5 + heatmap_colored * 0.5).astype(np.uint8)
            logger.debug("Blended heatmap with original image")

            # Metadata
            metadata = {
                "mean_abs_shap": float(np.abs(shap_vals_aggregated).mean()),
                "max_shap": float(shap_vals_aggregated.max()),
                "min_shap": float(shap_vals_aggregated.min()),
                "positive_contribution_ratio": float((shap_vals_aggregated > 0).sum() / shap_vals_aggregated.size),
                "explainer_type": "KernelExplainer",
                "num_superpixels": num_superpixels,
                "nsamples": 500
            }

            # Convert to base64
            visualization_base64 = self.array_to_base64(blended)

            logger.info(f"SHAP explanation complete: mean_abs={metadata['mean_abs_shap']:.4f}, "
                       f"range=[{metadata['min_shap']:.4f}, {metadata['max_shap']:.4f}]")

            return visualization_base64, metadata

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise Exception(f"SHAP explanation failed: {str(e)}")
