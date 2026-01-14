"""
Image processing utilities for medical images (X-rays)
"""

import numpy as np
from PIL import Image
import cv2


class ImageProcessor:
    """Processes medical images for model input"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize image processor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def process(self, image_path):
        """
        Process image file for model input
        
        Args:
            image_path: Path to image file (.jpg, .png, .jpeg)
            
        Returns:
            numpy array: Processed image normalized to 0-1
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
    
    def preprocess_for_model(self, image_array, model_name=None):
        """
        Apply model-specific preprocessing
        
        Args:
            image_array: Input image array
            model_name: Name of the model (for specific preprocessing)
            
        Returns:
            numpy array: Preprocessed image
        """
        # Ensure proper shape
        if len(image_array.shape) == 3:
            # Already has channels
            processed = image_array
        else:
            # Add channels if grayscale
            processed = np.stack([image_array] * 3, axis=-1)
        
        # Model-specific preprocessing can be added here
        if model_name and 'VGG' in model_name.upper():
            # VGG expects mean-centered images
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            processed = (processed - mean) / std
        
        return processed
    
    def enhance_contrast(self, image_array):
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image_array: Input image array (0-1 normalized)
            
        Returns:
            numpy array: Contrast-enhanced image
        """
        # Convert to uint8
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = np.zeros_like(img_uint8)
        
        for i in range(3):
            enhanced[:, :, i] = clahe.apply(img_uint8[:, :, i])
        
        # Convert back to float
        return enhanced.astype(np.float32) / 255.0
    
    def augment(self, image_array, horizontal_flip=False, vertical_flip=False, rotation=0):
        """
        Apply data augmentation
        
        Args:
            image_array: Input image
            horizontal_flip: Apply horizontal flip
            vertical_flip: Apply vertical flip
            rotation: Rotation angle in degrees
            
        Returns:
            numpy array: Augmented image
        """
        img = image_array.copy()
        
        if horizontal_flip:
            img = np.fliplr(img)
        
        if vertical_flip:
            img = np.flipud(img)
        
        if rotation != 0:
            # Convert to uint8 for rotation
            img_uint8 = (img * 255).astype(np.uint8)
            center = tuple(np.array(img.shape[:2]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img_rotated = cv2.warpAffine(img_uint8, rot_mat, img.shape[:2])
            img = img_rotated.astype(np.float32) / 255.0
        
        return img
