"""
Model loading utilities for audio and image classification models
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
import streamlit as st


class ModelLoader:
    """Loads trained models for classification"""
    
    def __init__(self):
        """Initialize model loader"""
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.loaded_models = {}  # Cache loaded models
    
    def load_model(self, model_name, task_type):
        """
        Load a trained model
        
        Args:
            model_name: Name of the model architecture
            task_type: 'audio' or 'image'
            
        Returns:
            Loaded Keras model
        """
        cache_key = f"{task_type}_{model_name}"
        
        # Return cached model if available
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Try to load from saved models
        model_path = self.models_dir / task_type / model_name / 'model.h5'
        
        if model_path.exists():
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                self.loaded_models[cache_key] = model
                return model
            except Exception as e:
                st.warning(f"Could not load saved model from {model_path}: {e}")
        
        # If no saved model, create a new one (for demonstration)
        st.info(f"Creating new {model_name} model for {task_type} classification")
        model = self._create_model(model_name, task_type)
        self.loaded_models[cache_key] = model
        
        return model
    
    def _create_model(self, model_name, task_type):
        """
        Create a model architecture
        
        Args:
            model_name: Name of the model architecture
            task_type: 'audio' or 'image'
            
        Returns:
            Compiled Keras model
        """
        input_shape = (224, 224, 3)  # Standard input size for both audio spectrograms and images
        
        # Base models from Keras applications
        if model_name.upper() == 'VGG16':
            base_model = tf.keras.applications.VGG16(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
        elif model_name.upper() == 'RESNET50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
        elif model_name.upper() == 'INCEPTIONV3':
            input_shape = (299, 299, 3)  # InceptionV3 uses different input size
            base_model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
        elif model_name.upper() == 'MOBILENET':
            base_model = tf.keras.applications.MobileNet(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
        elif model_name.upper() == 'DENSENET':
            base_model = tf.keras.applications.DenseNet121(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
        elif model_name.upper() == 'ALEXNET':
            # AlexNet-like architecture (not in Keras applications)
            base_model = self._create_alexnet(input_shape)
        else:
            # Default to simple CNN
            base_model = self._create_simple_cnn(input_shape)
        
        # Add classification head
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        
        # Add dense layers
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _create_alexnet(self, input_shape):
        """
        Create AlexNet-like architecture
        
        Args:
            input_shape: Input shape tuple
            
        Returns:
            Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((3, 3), strides=2),
            tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((3, 3), strides=2),
            tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((3, 3), strides=2),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='alexnet_base')
        
        return model
    
    def _create_simple_cnn(self, input_shape):
        """
        Create a simple CNN architecture
        
        Args:
            input_shape: Input shape tuple
            
        Returns:
            Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ], name='simple_cnn_base')
        
        return model
    
    def get_available_models(self, task_type):
        """
        Get list of available models for a task type
        
        Args:
            task_type: 'audio' or 'image'
            
        Returns:
            List of model names
        """
        models_path = self.models_dir / task_type
        
        if not models_path.exists():
            # Return default models if directory doesn't exist
            if task_type == 'audio':
                return ['VGG16', 'MobileNet', 'ResNet50', 'InceptionV3']
            else:
                return ['VGG16', 'DenseNet', 'AlexNet']
        
        # Get directories in models path
        available = [d.name for d in models_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Return defaults if no models found
        if not available:
            if task_type == 'audio':
                return ['VGG16', 'MobileNet', 'ResNet50', 'InceptionV3']
            else:
                return ['VGG16', 'DenseNet', 'AlexNet']
        
        return available
    
    def save_model(self, model, model_name, task_type):
        """
        Save a trained model
        
        Args:
            model: Keras model to save
            model_name: Name of the model
            task_type: 'audio' or 'image'
        """
        save_dir = self.models_dir / task_type / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / 'model.h5'
        model.save(str(model_path))
        
        st.success(f"Model saved to {model_path}")
