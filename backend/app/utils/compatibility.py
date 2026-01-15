AUDIO_MODELS = {
    "VGG16": {
        "name": "VGG16",
        "type": "audio",
        "description": "VGG16 transfer learning for deepfake audio detection",
        "input_format": "wav",
        "classes": ["fake", "real"],
        "accuracy": 0.8392,
        "path": "audio/vgg16_model.keras"
    },
    "ResNet50": {
        "name": "ResNet50",
        "type": "audio",
        "description": "ResNet50 transfer learning for audio classification",
        "input_format": "wav",
        "classes": ["fake", "real"],
        "accuracy": 0.5239,
        "path": "audio/resnet50_model.keras"
    },
    "Custom_CNN": {
        "name": "Custom CNN",
        "type": "audio",
        "description": "Custom CNN architecture from scratch",
        "input_format": "wav",
        "classes": ["fake", "real"],
        "accuracy": 0.5037,
        "path": "audio/custom_cnn_model.keras"
    }
}

IMAGE_MODELS = {
    "VGG16_Image": {
        "name": "VGG16",
        "type": "image",
        "description": "VGG16 for lung cancer detection in chest X-rays",
        "input_format": "image",
        "classes": ["benign", "malignant"],
        "accuracy": 0.85,
        "path": "image/vgg16_without_vae_model.keras"
    },
    "DenseNet121": {
        "name": "DenseNet121",
        "type": "image",
        "description": "DenseNet121 for chest X-ray classification",
        "input_format": "image",
        "classes": ["benign", "malignant"],
        "accuracy": 0.88,
        "path": "image/densenet_whithout_vae_model.keras"
    }
}

# XAI method definitions
XAI_METHODS = {
    "LIME": {
        "name": "LIME",
        "full_name": "Local Interpretable Model-agnostic Explanations",
        "compatible_with": ["audio", "image"],
        "description": "Explains predictions by approximating the model locally with an interpretable model using superpixel perturbations",
        "color_scheme": "green/red overlay"
    },
    "SHAP": {
        "name": "SHAP",
        "full_name": "SHapley Additive exPlanations",
        "compatible_with": ["audio", "image"],
        "description": "Uses game theory to explain predictions by computing the contribution of each feature using Shapley values",
        "color_scheme": "red/blue heatmap"
    },
    "Grad-CAM": {
        "name": "Grad-CAM",
        "full_name": "Gradient-weighted Class Activation Mapping",
        "compatible_with": ["audio", "image"],  # Works on spectrograms too
        "description": "Visualizes important regions in CNN activations using gradients to highlight areas influencing predictions",
        "color_scheme": "red/yellow heatmap"
    }
}

# File type validation
ALLOWED_EXTENSIONS = {
    "audio": [".wav"],
    "image": [".jpg", ".jpeg", ".png"]
}

MAX_FILE_SIZE = {
    "audio": 50 * 1024 * 1024,  # 50 MB
    "image": 10 * 1024 * 1024   # 10 MB
}

MIME_TYPES = {
    "audio": ["audio/wav", "audio/x-wav", "audio/wave"],
    "image": ["image/jpeg", "image/png", "image/jpg"]
}


def get_compatible_models(file_type: str) -> list:
    if file_type == "audio":
        return list(AUDIO_MODELS.values())
    elif file_type == "image":
        return list(IMAGE_MODELS.values())
    else:
        return []

def get_compatible_xai_methods(file_type: str) -> list:
    compatible = []
    for method in XAI_METHODS.values():
        if file_type in method["compatible_with"]:
            compatible.append(method)
    return compatible

def get_model_by_name(model_name: str, file_type: str):
    models = AUDIO_MODELS if file_type == "audio" else IMAGE_MODELS
    if model_name in models:
        return models[model_name]
    for key, model in models.items():
        if model["name"] == model_name:
            return model
    return None

def get_xai_method_by_name(method_name: str):
    return XAI_METHODS.get(method_name)

def validate_file_extension(filename: str, file_type: str) -> bool:
    import os
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS.get(file_type, [])

def validate_mime_type(content_type: str, file_type: str) -> bool:
    return content_type in MIME_TYPES.get(file_type, [])
