"""
Mock data generators for classification results and XAI visualizations.
Used for UI development before real model integration.
"""

import random
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2


def generate_mock_classification(file_type: str, model_name: str) -> dict:
    """
    Generate mock classification results.

    Args:
        file_type: Either 'audio' or 'image'
        model_name: Name of the model

    Returns:
        Dictionary with prediction results
    """
    # Determine classes based on file type
    if file_type == "audio":
        classes = ["fake", "real"]
    else:  # image
        classes = ["benign", "malignant"]

    # Randomly select predicted class
    predicted_class = random.choice(classes)
    predicted_idx = classes.index(predicted_class)

    # Generate confidence (biased towards high confidence for better demo)
    confidence = random.uniform(0.75, 0.98)

    # Generate probabilities
    if predicted_idx == 0:
        prob_class_0 = confidence
        prob_class_1 = 1 - confidence
    else:
        prob_class_0 = 1 - confidence
        prob_class_1 = confidence

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            classes[0]: round(prob_class_0, 4),
            classes[1]: round(prob_class_1, 4)
        }
    }


def generate_mock_lime_visualization(input_image_path: str) -> tuple[str, dict]:
    """
    Generate mock LIME (superpixel) visualization.

    Args:
        input_image_path: Path to the input image

    Returns:
        Tuple of (base64_image, metadata)
    """
    # Load original image
    img = Image.open(input_image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Create superpixel overlay
    overlay = img_array.copy()

    # Generate random superpixel regions
    num_superpixels = random.randint(40, 60)

    for _ in range(num_superpixels):
        # Random superpixel position and size
        x = random.randint(0, 200)
        y = random.randint(0, 200)
        w = random.randint(15, 35)
        h = random.randint(15, 35)

        # Random importance (positive or negative)
        importance = random.uniform(-0.5, 0.5)

        # Color based on importance: green for positive, red for negative
        if importance > 0:
            color = np.array([0, int(255 * abs(importance)), 0])  # Green
        else:
            color = np.array([int(255 * abs(importance)), 0, 0])  # Red

        # Apply colored overlay to region
        overlay[y:y+h, x:x+w] = (overlay[y:y+h, x:x+w] * 0.6 + color * 0.4).astype(np.uint8)

    # Convert to PIL Image
    result_img = Image.fromarray(overlay)

    # Convert to base64
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Generate metadata
    metadata = {
        "num_superpixels": num_superpixels,
        "explanation_fit": round(random.uniform(0.75, 0.95), 2),
        "top_features": [
            {"superpixel_id": i, "weight": round(random.uniform(-0.5, 0.5), 3)}
            for i in range(5)
        ]
    }

    return f"data:image/png;base64,{img_base64}", metadata


def generate_mock_shap_visualization(input_image_path: str) -> tuple[str, dict]:
    """
    Generate mock SHAP (pixel attribution) visualization.

    Args:
        input_image_path: Path to the input image

    Returns:
        Tuple of (base64_image, metadata)
    """
    # Load original image
    img = Image.open(input_image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Create attribution heatmap
    # Generate random attribution values (centered around 0)
    attribution = np.random.randn(224, 224) * 0.1

    # Apply Gaussian blur for smooth heatmap
    attribution = cv2.GaussianBlur(attribution, (15, 15), 0)

    # Normalize to [-1, 1]
    attr_min = attribution.min()
    attr_max = attribution.max()
    if attr_max - attr_min > 0:
        attribution = 2 * (attribution - attr_min) / (attr_max - attr_min) - 1

    # Create colormap: red for positive, blue for negative
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            val = attribution[i, j]
            if val > 0:  # Positive attribution - red
                heatmap[i, j] = [int(255 * val), 0, 0]
            else:  # Negative attribution - blue
                heatmap[i, j] = [0, 0, int(255 * abs(val))]

    # Blend with original image
    blended = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    # Convert to PIL Image
    result_img = Image.fromarray(blended.astype(np.uint8))

    # Convert to base64
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Generate metadata
    metadata = {
        "mean_abs_shap": round(np.abs(attribution).mean(), 4),
        "max_attribution": round(attribution.max(), 4),
        "min_attribution": round(attribution.min(), 4),
        "positive_ratio": round((attribution > 0).sum() / attribution.size, 2)
    }

    return f"data:image/png;base64,{img_base64}", metadata


def generate_mock_gradcam_visualization(input_image_path: str) -> tuple[str, dict]:
    """
    Generate mock Grad-CAM (attention heatmap) visualization.

    Args:
        input_image_path: Path to the input image

    Returns:
        Tuple of (base64_image, metadata)
    """
    # Load original image
    img = Image.open(input_image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Create attention heatmap
    # Generate random center of attention
    center_x = random.randint(80, 144)
    center_y = random.randint(80, 144)

    # Create Gaussian-like attention map
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for i in range(224):
        for j in range(224):
            # Distance from center
            dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            # Gaussian-like decay
            heatmap[i, j] = np.exp(-dist ** 2 / (2 * 40 ** 2))

    # Add some noise
    heatmap += np.random.randn(224, 224) * 0.05
    heatmap = np.clip(heatmap, 0, 1)

    # Apply colormap (red to yellow)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    blended = cv2.addWeighted(img_array, 0.5, heatmap_colored, 0.5, 0)

    # Convert to PIL Image
    result_img = Image.fromarray(blended.astype(np.uint8))

    # Convert to base64
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Generate metadata
    metadata = {
        "target_layer": random.choice(["block5_conv3", "block4_conv2", "conv5_block3_out"]),
        "max_activation": round(heatmap.max(), 4),
        "mean_activation": round(heatmap.mean(), 4),
        "activation_coverage": round((heatmap > 0.3).sum() / heatmap.size, 2)
    }

    return f"data:image/png;base64,{img_base64}", metadata


def generate_mock_xai_visualization(input_image_path: str, xai_method: str) -> tuple[str, dict]:
    """
    Generate mock XAI visualization based on method.

    Args:
        input_image_path: Path to the input image
        xai_method: Name of XAI method (LIME, SHAP, or Grad-CAM)

    Returns:
        Tuple of (base64_image, metadata)
    """
    if xai_method == "LIME":
        return generate_mock_lime_visualization(input_image_path)
    elif xai_method == "SHAP":
        return generate_mock_shap_visualization(input_image_path)
    elif xai_method == "Grad-CAM":
        return generate_mock_gradcam_visualization(input_image_path)
    else:
        raise ValueError(f"Unknown XAI method: {xai_method}")


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image string with data URI prefix
    """
    with open(image_path, "rb") as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()

    # Determine MIME type
    if image_path.lower().endswith(".png"):
        mime = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    else:
        mime = "image/png"

    return f"data:{mime};base64,{img_base64}"
