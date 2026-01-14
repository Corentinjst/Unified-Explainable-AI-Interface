"""
LIME (Local Interpretable Model-agnostic Explanations) explainer for images.
"""

import numpy as np
import logging
from typing import Tuple, Dict
from lime import lime_image
from skimage.segmentation import mark_boundaries

from .base_explainer import BaseExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME explainer for image classification models.

    LIME explains predictions by approximating the model locally with an
    interpretable model using superpixel perturbations. It identifies which
    regions (superpixels) of the image contribute most to the prediction.
    """

    def __init__(self, num_samples: int = 1000):
        """
        Initialize LIME explainer.

        Args:
            num_samples: Number of perturbations to generate (default 1000)
                        Higher = more accurate but slower
        """
        self.explainer = lime_image.LimeImageExplainer()
        self.num_samples = num_samples
        logger.info(f"LIME explainer initialized with {num_samples} samples")

    def explain(
        self,
        input_data: np.ndarray,
        model,
        predicted_class: int
    ) -> Tuple[str, Dict]:
        """
        Generate LIME explanation using superpixel perturbations.

        Process:
        1. Create prediction wrapper that handles model input
        2. Generate explanation using LIME
        3. Extract top features (superpixels)
        4. Create visualization with marked boundaries
        5. Return as base64 + metadata

        Args:
            input_data: Image array (224, 224, 3) with values [0, 255]
            model: TensorFlow model
            predicted_class: Predicted class index (0 or 1)

        Returns:
            Tuple of (visualization_base64, metadata)
        """
        try:
            logger.info("Generating LIME explanation...")

            # Define prediction wrapper
            # LIME will pass batches of perturbed images
            def predict_fn(images):
                """
                Prediction function for LIME.

                Args:
                    images: numpy array of shape (n_samples, 224, 224, 3) with values [0, 255]

                Returns:
                    numpy array of shape (n_samples, 2) with class probabilities
                """
                # Model expects [0, 255] (has Rescaling layer)
                # Model outputs (n_samples, 1) for binary classification
                predictions = model.predict(images, verbose=0)

                # Convert to (n_samples, 2) format for LIME
                # [prob_class_0, prob_class_1]
                prob_class_1 = predictions.flatten()
                prob_class_0 = 1 - prob_class_1
                return np.column_stack([prob_class_0, prob_class_1])

            # Generate explanation
            logger.debug(f"Running LIME with {self.num_samples} samples...")
            explanation = self.explainer.explain_instance(
                input_data.astype(np.float64),  # LIME expects float
                predict_fn,
                top_labels=2,  # Explain top 2 classes
                hide_color=0,  # Color for hiding regions
                num_samples=self.num_samples
            )
            logger.debug("LIME explanation generated")

            # Get image and mask for predicted class
            temp, mask = explanation.get_image_and_mask(
                predicted_class,
                positive_only=False,  # Show both positive and negative features
                num_features=10,  # Number of features to highlight
                hide_rest=False  # Show full image
            )
            logger.debug(f"Extracted mask with {len(np.unique(mask))} unique segments")

            # Create visualization with marked boundaries
            # mark_boundaries expects values in [0, 1]
            visualization = mark_boundaries(
                temp / 255.0,
                mask,
                color=(0, 1, 0),  # Green boundaries
                mode='thick'
            )
            # Convert back to [0, 255]
            visualization = (visualization * 255).astype(np.uint8)
            logger.debug("Visualization created with marked boundaries")

            # Extract metadata
            # Get feature weights for the predicted class
            feature_weights = explanation.local_exp[predicted_class]
            top_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)[:5]

            metadata = {
                "num_features": len(feature_weights),
                "num_superpixels": len(np.unique(mask)),
                "num_samples": self.num_samples,
                "top_features": [
                    {
                        "feature_id": int(feat),
                        "weight": float(weight),
                        "importance": "positive" if weight > 0 else "negative"
                    }
                    for feat, weight in top_features
                ]
            }

            # Convert to base64
            visualization_base64 = self.array_to_base64(visualization)

            logger.info(f"LIME explanation complete: {len(feature_weights)} features, "
                       f"{len(np.unique(mask))} superpixels")

            return visualization_base64, metadata

        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise Exception(f"LIME explanation failed: {str(e)}")
