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

        Process:
        1. Find last convolutional layer automatically
        2. Create gradient model (conv outputs + predictions)
        3. Compute gradients of predicted class w.r.t. conv outputs
        4. Weight conv activations by gradients (global average pooling)
        5. Create heatmap and apply ReLU
        6. Resize to input size and apply colormap
        7. Overlay on original image
        8. Return as base64 + metadata

        Args:
            input_data: Image array (224, 224, 3) with values [0, 255]
            model: TensorFlow model
            predicted_class: Predicted class index (0 or 1)

        Returns:
            Tuple of (visualization_base64, metadata)
        """
        try:
            logger.info("Generating Grad-CAM explanation...")

            # Find last convolutional layer
            last_conv_layer_name = self._find_last_conv_layer(model)
            logger.debug(f"Using convolutional layer: {last_conv_layer_name}")

            # Create gradient model
            # This outputs both the conv layer activations and the final prediction
            last_conv_layer = model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output]
            )
            logger.debug("Gradient model created")

            # Prepare input
            input_batch = np.expand_dims(input_data, axis=0)  # (1, 224, 224, 3)
            input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)

            # Compute gradients using GradientTape
            logger.debug("Computing gradients...")
            with tf.GradientTape() as tape:
                # Forward pass
                conv_outputs, predictions = grad_model(input_tensor)

                # For binary classification, get the prediction for our class
                # predictions shape: (1, 1) - sigmoid output
                loss = predictions[:, 0]

            # Get gradients of the loss w.r.t. conv outputs
            grads = tape.gradient(loss, conv_outputs)
            logger.debug(f"Gradients computed: shape={grads.shape}")

            # Global average pooling of gradients (weights)
            # This gives importance of each feature map
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            logger.debug(f"Pooled gradients: shape={pooled_grads.shape}")

            # Weight conv outputs by gradients
            conv_outputs = conv_outputs[0].numpy()  # Remove batch dimension
            pooled_grads = pooled_grads.numpy()

            # Multiply each feature map by its importance weight
            for i in range(len(pooled_grads)):
                conv_outputs[:, :, i] *= pooled_grads[i]

            # Create heatmap (average across all feature maps)
            heatmap = np.mean(conv_outputs, axis=-1)
            logger.debug(f"Heatmap created: shape={heatmap.shape}, "
                        f"range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")

            # Apply ReLU (only positive contributions)
            heatmap = np.maximum(heatmap, 0)

            # Normalize to [0, 1]
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            logger.debug("Heatmap normalized")

            # Resize heatmap to input size
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            # Apply colormap (COLORMAP_JET: red/yellow for high activation)
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            # Convert BGR to RGB
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            logger.debug("Applied JET colormap")

            # Overlay on original image (50% opacity each)
            blended = (input_data * 0.5 + heatmap_colored * 0.5).astype(np.uint8)
            logger.debug("Blended heatmap with original image")

            # Metadata
            metadata = {
                "target_layer": last_conv_layer_name,
                "max_activation": float(heatmap_resized.max()),
                "mean_activation": float(heatmap_resized.mean()),
                "activation_coverage": float((heatmap_resized > 0.3).sum() / heatmap_resized.size)
            }

            # Convert to base64
            visualization_base64 = self.array_to_base64(blended)

            logger.info(f"Grad-CAM explanation complete: layer={last_conv_layer_name}, "
                       f"max_activation={metadata['max_activation']:.4f}")

            return visualization_base64, metadata

        except Exception as e:
            logger.error(f"Error generating Grad-CAM explanation: {str(e)}")
            raise Exception(f"Grad-CAM explanation failed: {str(e)}")

    @staticmethod
    def _find_last_conv_layer(model) -> str:
        """
        Find the last convolutional layer in the model.

        Searches backwards through the model layers to find the last
        layer with "conv" in its name.

        Args:
            model: TensorFlow model

        Returns:
            Name of the last convolutional layer

        Raises:
            ValueError: If no convolutional layer is found
        """
        # Search backwards through layers
        for layer in reversed(model.layers):
            # Check if layer name contains "conv" (case insensitive)
            if 'conv' in layer.name.lower():
                logger.debug(f"Found last conv layer: {layer.name}")
                return layer.name

        # If no conv layer found, raise error
        raise ValueError(
            "No convolutional layer found in model. "
            "Grad-CAM requires a CNN architecture with convolutional layers."
        )
