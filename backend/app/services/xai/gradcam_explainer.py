import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, Dict

from .base_explainer import BaseExplainer

class GradCAMExplainer(BaseExplainer):
    def __init__(self):
        pass

    def explain(self, input_data: np.ndarray, model, predicted_class: int) -> Tuple[str, Dict]:
        input_batch = np.expand_dims(input_data, axis=0)
        input_tensor = tf.convert_to_tensor(input_batch, dtype=tf.float32)
        last_conv_layer, last_conv_layer_name = self._find_last_conv_layer(model)
        conv_outputs, grads = self._compute_gradcam_gradients(model, input_tensor, last_conv_layer)

        if grads is None:
            raise ValueError("Could not compute gradients")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        for i in range(len(pooled_grads_np)):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]

        heatmap = np.mean(conv_outputs_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blended = (input_data * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

        metadata = {
            "target_layer": last_conv_layer_name,
            "max_activation": float(heatmap_resized.max()),
            "mean_activation": float(heatmap_resized.mean()),
            "activation_coverage": float((heatmap_resized > 0.3).sum() / heatmap_resized.size)
        }

        visualization_base64 = self.array_to_base64(blended)
        return visualization_base64, metadata

    def _compute_gradcam_gradients(self, model, input_tensor, target_layer):
        conv_output_value = None
        original_call = target_layer.call

        def capturing_call(inputs, *args, **kwargs):
            nonlocal conv_output_value
            output = original_call(inputs, *args, **kwargs)
            conv_output_value = output
            return output

        target_layer.call = capturing_call

        try:
            with tf.GradientTape() as tape:
                predictions = model(input_tensor, training=False)
                tape.watch(conv_output_value)
                loss = predictions[:, 0]
            grads = tape.gradient(loss, conv_output_value)
            return conv_output_value, grads
        finally:
            target_layer.call = original_call

    def _find_last_conv_layer(self, model) -> Tuple[tf.keras.layers.Layer, str]:
        last_conv_layer = None
        last_conv_name = None

        def search_layers(layers, prefix=""):
            nonlocal last_conv_layer, last_conv_name
            for layer in layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    last_conv_name = prefix + layer.name
                elif hasattr(layer, 'layers'):
                    search_layers(layer.layers, prefix + layer.name + "/")

        search_layers(model.layers)
        if last_conv_layer is None:
            raise ValueError("No conv layer found")
        return last_conv_layer, last_conv_name
