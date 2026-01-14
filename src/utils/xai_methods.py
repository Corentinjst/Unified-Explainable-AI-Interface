"""
XAI (Explainable AI) methods implementation
Includes LIME, SHAP, and Grad-CAM
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

# XAI libraries
try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
except ImportError:
    lime_image = None

try:
    import shap
except ImportError:
    shap = None


class XAIExplainer:
    """Generates explanations for model predictions using various XAI techniques"""
    
    def __init__(self, model, task_type):
        """
        Initialize XAI explainer
        
        Args:
            model: Trained Keras model
            task_type: 'audio' or 'image'
        """
        self.model = model
        self.task_type = task_type
    
    def explain(self, input_data, method, model_name):
        """
        Generate explanation for the input using specified method
        
        Args:
            input_data: Preprocessed input (numpy array)
            method: XAI method ('LIME', 'SHAP', 'Grad-CAM')
            model_name: Name of the model
            
        Returns:
            PIL Image: Explanation visualization
        """
        if method.upper() == 'LIME':
            return self._explain_lime(input_data)
        elif method.upper() == 'SHAP':
            return self._explain_shap(input_data)
        elif method.upper() == 'GRAD-CAM':
            return self._explain_gradcam(input_data, model_name)
        else:
            return None
    
    def _explain_lime(self, input_data):
        """
        Generate LIME explanation
        
        Args:
            input_data: Input image (numpy array, normalized 0-1)
            
        Returns:
            PIL Image: LIME visualization
        """
        if lime_image is None:
            return self._create_error_image("LIME not installed. Run: pip install lime")
        
        try:
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Define prediction function
            def predict_fn(images):
                # Ensure correct shape
                preds = self.model.predict(images, verbose=0)
                # For binary classification, return probabilities for both classes
                if len(preds.shape) == 2 and preds.shape[1] == 1:
                    return np.hstack([1 - preds, preds])
                return preds
            
            # Generate explanation
            explanation = explainer.explain_instance(
                input_data,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
                segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
            )
            
            # Get mask for top prediction
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=10,
                hide_rest=False
            )
            
            # Create visualization
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax[0].imshow(input_data)
            ax[0].set_title('Original Input')
            ax[0].axis('off')
            
            # LIME explanation overlay
            ax[1].imshow(input_data)
            ax[1].imshow(mask, cmap='RdYlGn', alpha=0.5)
            ax[1].set_title('LIME Explanation')
            ax[1].axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
        
        except Exception as e:
            return self._create_error_image(f"LIME error: {str(e)}")
    
    def _explain_shap(self, input_data):
        """
        Generate SHAP explanation
        
        Args:
            input_data: Input image (numpy array)
            
        Returns:
            PIL Image: SHAP visualization
        """
        if shap is None:
            return self._create_error_image("SHAP not installed. Run: pip install shap")
        
        try:
            # Create a background dataset (using the input itself for simplicity)
            background = np.expand_dims(input_data, axis=0)
            
            # Create SHAP explainer
            explainer = shap.GradientExplainer(self.model, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(np.expand_dims(input_data, axis=0))
            
            # Handle different output shapes
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get the SHAP values for the first (and only) sample
            shap_vals = shap_values[0]
            
            # Sum across color channels if needed
            if len(shap_vals.shape) == 3 and shap_vals.shape[-1] == 3:
                shap_vals = np.mean(shap_vals, axis=-1)
            
            # Create visualization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax[0].imshow(input_data)
            ax[0].set_title('Original Input')
            ax[0].axis('off')
            
            # SHAP values heatmap
            im = ax[1].imshow(shap_vals, cmap='RdBu', vmin=-np.max(np.abs(shap_vals)), vmax=np.max(np.abs(shap_vals)))
            ax[1].set_title('SHAP Values')
            ax[1].axis('off')
            plt.colorbar(im, ax=ax[1])
            
            # Overlay
            ax[2].imshow(input_data)
            ax[2].imshow(np.abs(shap_vals), cmap='hot', alpha=0.5)
            ax[2].set_title('SHAP Overlay')
            ax[2].axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
        
        except Exception as e:
            return self._create_error_image(f"SHAP error: {str(e)}")
    
    def _explain_gradcam(self, input_data, model_name):
        """
        Generate Grad-CAM explanation
        
        Args:
            input_data: Input image (numpy array)
            model_name: Name of the model (to find conv layer)
            
        Returns:
            PIL Image: Grad-CAM visualization
        """
        try:
            # Find the last convolutional layer
            conv_layer = self._find_last_conv_layer(self.model)
            
            if conv_layer is None:
                return self._create_error_image("No convolutional layer found in model")
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=[self.model.input],
                outputs=[conv_layer.output, self.model.output]
            )
            
            # Compute gradients
            input_tensor = tf.expand_dims(input_data, axis=0)
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor)
                # For binary classification, use the single output
                if predictions.shape[-1] == 1:
                    loss = predictions[:, 0]
                else:
                    loss = predictions[:, tf.argmax(predictions[0])]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Compute guided gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by gradient importance
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to match input size
            heatmap = cv2.resize(heatmap, (input_data.shape[1], input_data.shape[0]))
            
            # Create visualization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax[0].imshow(input_data)
            ax[0].set_title('Original Input')
            ax[0].axis('off')
            
            # Heatmap
            im = ax[1].imshow(heatmap, cmap='jet')
            ax[1].set_title('Grad-CAM Heatmap')
            ax[1].axis('off')
            plt.colorbar(im, ax=ax[1])
            
            # Overlay
            ax[2].imshow(input_data)
            ax[2].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[2].set_title('Grad-CAM Overlay')
            ax[2].axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
        
        except Exception as e:
            return self._create_error_image(f"Grad-CAM error: {str(e)}")
    
    def _find_last_conv_layer(self, model):
        """
        Find the last convolutional layer in the model
        
        Args:
            model: Keras model
            
        Returns:
            Layer: Last convolutional layer
        """
        # Try to find in main model
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer
            # Check if layer is a model (for functional API)
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower():
                        return sublayer
        
        return None
    
    def _create_error_image(self, error_message):
        """
        Create an error image with message
        
        Args:
            error_message: Error message to display
            
        Returns:
            PIL Image: Error image
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error:\n{error_message}", 
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
