"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainer for CNNs.
"""

import tensorflow as tf
import numpy as np
import cv2
import logging
from typing import Tuple, Dict

from .base_explainer import BaseExplainer

logger = logging.getLogger(__name__)


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM explainer for convolutional neural networks.

    Grad-CAM visualizes important regions in CNN activations using gradients
    to highlight areas that most influence the model's predictions. It creates
    a heatmap showing where the model is "looking" when making decisions.
    """

    def __init__(self):
        """Initialize Grad-CAM explainer."""
        logger.info("Grad-CAM explainer initialized")

    def explain(
        self,
        input_data: np.ndarray,
        model,
        predicted_class: int
    ) -> Tuple[str, Dict]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_data: Image array (224, 224, 3) with values [0, 255]
            model: TensorFlow model
            predicted_class: Predicted class index (0 or 1)

        Returns:
            Tuple of (visualization_base64, metadata)
        """
        try:
            logger.info("Generating Grad-CAM explanation...")

            # Prepare input
            input_batch = np.expand_dims(input_data, axis=0)  # (1, 224, 224, 3)
            input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)

            # Find the base CNN model and last conv layer
            base_model, last_conv_layer_name = self._find_base_model_and_conv(model)
            logger.debug(f"Using base model: {base_model.name}, conv layer: {last_conv_layer_name}")

            # Create gradient model from the base model
            last_conv_layer = base_model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=[last_conv_layer.output, base_model.output]
            )

            # Compute gradients
            logger.debug("Computing gradients...")
            with tf.GradientTape() as tape:
                # Forward pass through base model
                conv_outputs, base_output = grad_model(input_tensor)

                # Forward pass through the rest of the model to get final prediction
                # Find layers after base model and apply them
                x = base_output
                found_base = False
                for layer in model.layers:
                    if layer.name == base_model.name:
                        found_base = True
                        continue
                    if found_base:
                        x = layer(x)

                predictions = x

                # For binary classification, get the prediction
                loss = predictions[:, 0]

            # Get gradients of the loss w.r.t. conv outputs
            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                raise ValueError("Could not compute gradients.")

            logger.debug(f"Gradients computed: shape={grads.shape}")

            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight conv outputs by gradients
            conv_outputs_np = conv_outputs[0].numpy()
            pooled_grads_np = pooled_grads.numpy()

            for i in range(len(pooled_grads_np)):
                conv_outputs_np[:, :, i] *= pooled_grads_np[i]

            # Create heatmap
            heatmap = np.mean(conv_outputs_np, axis=-1)
            heatmap = np.maximum(heatmap, 0)

            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # Resize and colorize
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Blend with original
            blended = (input_data * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

            metadata = {
                "target_layer": last_conv_layer_name,
                "max_activation": float(heatmap_resized.max()),
                "mean_activation": float(heatmap_resized.mean()),
                "activation_coverage": float((heatmap_resized > 0.3).sum() / heatmap_resized.size)
            }

            visualization_base64 = self.array_to_base64(blended)

            logger.info(f"Grad-CAM explanation complete: layer={last_conv_layer_name}")

            return visualization_base64, metadata

        except Exception as e:
            logger.error(f"Error generating Grad-CAM explanation: {str(e)}")
            raise Exception(f"Grad-CAM explanation failed: {str(e)}")

    def _find_base_model_and_conv(self, model) -> Tuple[tf.keras.Model, str]:
        """
        Find the base CNN model (e.g., VGG16) and its last conv layer.

        Args:
            model: TensorFlow model (may contain nested models)

        Returns:
            Tuple of (base_model, last_conv_layer_name)
        """
        # First, check if this model itself has Conv2D layers at the top level
        top_level_conv = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                top_level_conv = layer.name

        if top_level_conv:
            # Model has Conv2D at top level, use it directly
            logger.debug(f"Found top-level Conv2D: {top_level_conv}")
            return model, top_level_conv

        # Look for nested models that contain Conv2D layers
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                # This is a nested model (like VGG16, ResNet, etc.)
                last_conv = None
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv = sub_layer.name

                if last_conv:
                    logger.debug(f"Found nested model '{layer.name}' with conv layer: {last_conv}")
                    return layer, last_conv

        raise ValueError(
            "No convolutional layer found in model. "
            "Grad-CAM requires a CNN architecture with convolutional layers."
        )
