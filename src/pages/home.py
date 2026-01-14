import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import os

from utils.audio_processor import AudioProcessor
from utils.image_processor import ImageProcessor
from utils.model_loader import ModelLoader
from utils.xai_methods import XAIExplainer


def show():
    """Main home page for file upload and single prediction"""

    st.markdown('<p class="main-header">Unified Explainable AI Interface</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-modal Classification with Explainability (Audio & Image)</p>',
        unsafe_allow_html=True
    )

    # File upload section
    st.header("1. Upload Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload audio (.wav) or image (.jpg, .png, .jpeg)",
            type=['wav', 'jpg', 'jpeg', 'png'],
            help="Upload an audio file for deepfake detection or a chest X-ray for lung cancer detection"
        )

    with col2:
        if uploaded_file:
            file_type = _detect_file_type(uploaded_file.name)
            st.info(f"File type: **{file_type}**")

            if file_type == "Audio":
                st.success("Audio file detected - Deepfake detection available")
            else:
                st.success("Image file detected - Lung cancer detection available")

    if not uploaded_file:
        st.info("Please upload a file to begin")
        _show_sample_info()
        return

    # Detect file type
    file_type = _detect_file_type(uploaded_file.name)

    # Model selection
    st.header("2. Select Model")
    available_models = _get_available_models(file_type)

    if not available_models:
        st.error(f"No trained models found for {file_type} data. Please train models first.")
        return

    selected_model = st.selectbox(
        "Choose a classification model",
        available_models,
        help="Select the model architecture for classification"
    )

    # XAI method selection
    st.header("3. Select XAI Method")
    available_xai = _get_available_xai_methods(file_type)

    selected_xai = st.selectbox(
        "Choose an explainability method",
        available_xai,
        help="Select the XAI technique to visualize model decisions"
    )

    # Display compatibility info
    with st.expander("Why are some methods unavailable?"):
        st.write(f"""
        XAI methods are automatically filtered based on your input type (**{file_type}**).

        **Available for {file_type}:**
        {', '.join(available_xai)}

        **Note:** Some methods like Grad-CAM work best with convolutional models,
        while LIME and SHAP are more universal.
        """)

    # Analyze button
    st.header("4. Analyze")

    if st.button("Run Classification & Explanation", type="primary"):
        with st.spinner("Processing..."):
            result = _process_file(uploaded_file, file_type, selected_model, selected_xai)

            if result:
                _display_results(result, file_type)

                # Add to history for comparison
                st.session_state.results_history.append({
                    'file_name': uploaded_file.name,
                    'file_type': file_type,
                    'model': selected_model,
                    'xai_method': selected_xai,
                    'result': result
                })

                st.success(f"Results saved! You can compare {len(st.session_state.results_history)} results in the Comparison page.")


def _detect_file_type(filename):
    """Detect if file is audio or image"""
    ext = Path(filename).suffix.lower()
    if ext == '.wav':
        return "Audio"
    elif ext in ['.jpg', '.jpeg', '.png']:
        return "Image"
    return "Unknown"


def _get_available_models(file_type):
    """Get list of available models for the given file type"""
    models_dir = Path(__file__).parent.parent.parent / 'models'

    if file_type == "Audio":
        audio_models_dir = models_dir / 'audio'
        if audio_models_dir.exists():
            models = [d.name for d in audio_models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            return models if models else ["VGG16", "MobileNet", "ResNet50", "InceptionV3"]
        return ["VGG16", "MobileNet", "ResNet50", "InceptionV3"]

    elif file_type == "Image":
        image_models_dir = models_dir / 'image'
        if image_models_dir.exists():
            models = [d.name for d in image_models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            return models if models else ["AlexNet", "DenseNet", "VGG16"]
        return ["AlexNet", "DenseNet", "VGG16"]

    return []


def _get_available_xai_methods(file_type):
    """Get XAI methods compatible with the file type"""

    # All methods work with both types as they're both converted to images
    common_methods = ["LIME", "SHAP", "Grad-CAM"]

    if file_type == "Audio":
        # Audio (spectrograms) - all methods available
        return common_methods

    elif file_type == "Image":
        # Image (X-rays) - all methods available
        return common_methods

    return common_methods


def _process_file(uploaded_file, file_type, model_name, xai_method):
    """Process the uploaded file and generate predictions with explanations"""

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process based on file type
        if file_type == "Audio":
            processor = AudioProcessor()
            processed_input = processor.process(tmp_path)
            task_type = "audio"
        else:
            processor = ImageProcessor()
            processed_input = processor.process(tmp_path)
            task_type = "image"

        # Load model
        model_loader = ModelLoader()
        model = model_loader.load_model(model_name, task_type)

        # Make prediction
        prediction = model.predict(np.expand_dims(processed_input, axis=0))

        # Generate explanation
        explainer = XAIExplainer(model, task_type)
        explanation = explainer.explain(processed_input, xai_method, model_name)

        # Clean up temp file
        os.unlink(tmp_path)

        return {
            'processed_input': processed_input,
            'original_file': tmp_path,
            'prediction': prediction,
            'explanation': explanation,
            'xai_method': xai_method
        }

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def _display_results(result, file_type):
    """Display classification results and explanations"""

    st.header("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        if file_type == "Audio":
            st.image(result['processed_input'], caption="Audio Spectrogram", use_container_width=True)
        else:
            st.image(result['processed_input'], caption="X-Ray Image", use_container_width=True)

    with col2:
        st.subheader("Classification Result")

        prediction = result['prediction'][0]

        if file_type == "Audio":
            # Binary classification: fake (0) vs real (1)
            prob = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)

            if prob > 0.5:
                st.success(f"**REAL AUDIO** (Confidence: {prob*100:.2f}%)")
            else:
                st.error(f"**FAKE AUDIO** (Confidence: {(1-prob)*100:.2f}%)")

            # Progress bar
            st.progress(prob)

        else:
            # Binary classification: benign (0) vs malignant (1)
            prob = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)

            if prob > 0.5:
                st.error(f"**MALIGNANT** (Confidence: {prob*100:.2f}%)")
            else:
                st.success(f"**BENIGN** (Confidence: {(1-prob)*100:.2f}%)")

            # Progress bar
            st.progress(prob)

    # Explanation visualization
    st.subheader(f"XAI Explanation: {result['xai_method']}")

    if result['explanation'] is not None:
        st.image(result['explanation'], caption=f"{result['xai_method']} Visualization", use_container_width=True)

        with st.expander("Understanding the Explanation"):
            _show_xai_explanation(result['xai_method'], file_type)
    else:
        st.warning("Explanation visualization not available")


def _show_xai_explanation(xai_method, file_type):
    """Show explanation of what the XAI visualization means"""

    if xai_method == "LIME":
        st.write("""
        **LIME (Local Interpretable Model-agnostic Explanations)**

        - Highlights regions that most influenced the prediction
        - Green areas: regions that support the predicted class
        - Red areas: regions that argue against the predicted class
        - Works by perturbing the input and observing prediction changes
        """)

    elif xai_method == "SHAP":
        st.write("""
        **SHAP (SHapley Additive exPlanations)**

        - Shows the contribution of each feature/region to the prediction
        - Red areas: positive contribution to the predicted class
        - Blue areas: negative contribution to the predicted class
        - Based on game theory and Shapley values
        """)

    elif xai_method == "Grad-CAM":
        st.write("""
        **Grad-CAM (Gradient-weighted Class Activation Mapping)**

        - Heat map showing where the model "looked" to make its decision
        - Red/Yellow areas: regions the model focused on most
        - Blue/Purple areas: less important regions
        - Works by analyzing gradients flowing into the final conv layer
        """)


def _show_sample_info():
    """Show sample information about the system"""

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Audio: Deepfake Detection")
        st.write("""
        Upload a .wav audio file to detect if it's real or synthesized (deepfake).

        **Available Models:**
        - VGG16
        - MobileNet
        - ResNet50
        - InceptionV3

        **XAI Methods:**
        - LIME: Highlights influential audio features
        - SHAP: Shows feature importance
        - Grad-CAM: Visualizes attention regions
        """)

    with col2:
        st.subheader("Image: Lung Cancer Detection")
        st.write("""
        Upload a chest X-ray image (.jpg, .png) to detect lung lesions.

        **Available Models:**
        - AlexNet
        - DenseNet
        - VGG16

        **XAI Methods:**
        - LIME: Highlights suspicious regions
        - SHAP: Shows pixel importance
        - Grad-CAM: Visualizes model attention
        """)
