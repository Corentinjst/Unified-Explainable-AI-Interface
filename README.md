# Unified Explainable AI Interface

A Streamlit-based web application for multi-modal AI classification with explainability. This application integrates deepfake audio detection and lung cancer detection into a single interactive platform.

## Features

### Multi-Modal Classification
- **Audio Classification**: Deepfake detection for audio files (.wav)
- **Image Classification**: Lung lesion detection in chest X-rays (.jpg, .png, .jpeg)

### Explainable AI (XAI) Methods
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- **Grad-CAM** (Gradient-weighted Class Activation Mapping)

### Comparison Dashboard
- Side-by-side comparison of multiple XAI explanations
- Compare different models on the same input
- Compare different XAI methods for the same prediction

## Installation

1. Clone the repository:
```bash
cd Unified-Explainable-AI-Interface
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

#### Home Page
1. **Upload Data**: Drag and drop or browse for an audio (.wav) or image (.jpg, .png, .jpeg) file
2. **Select Model**: Choose from available classification models
   - Audio: VGG16, MobileNet, ResNet50, InceptionV3
   - Image: AlexNet, DenseNet, VGG16
3. **Select XAI Method**: Choose an explainability technique (LIME, SHAP, Grad-CAM)
4. **Analyze**: Click "Run Classification & Explanation" to see results

#### Comparison Page
1. Navigate to the "Comparison" page using the sidebar
2. Select 2-4 results from previous analyses
3. View side-by-side comparisons with insights and statistics

### Training Models

Before using the application with real predictions, you should train the models using the provided notebooks:

#### Audio Models (Deepfake Detection)
```bash
# First, convert audio to spectrograms
jupyter notebook notebooks/00_audio_to_spectrogram.ipynb

# Then train models
jupyter notebook notebooks/01_train_vgg16.ipynb
jupyter notebook notebooks/02_train_custom_cnn.ipynb
# ... and so on
```

#### Image Models (Lung Cancer Detection)
```bash
# Process the lung cancer dataset
jupyter notebook notebooks/06_processing_lung_cancer.ipynb

# Then train models
jupyter notebook notebooks/09_train_densenet_without_vae.ipynb
# ... and so on
```

Trained models will be saved to:
- `models/audio/` for audio classification models
- `models/image/` for image classification models

## Project Structure

```
Unified-Explainable-AI-Interface/
├── app.py                  # Main Streamlit application
├── src/
│   ├── pages/
│   │   ├── home.py        # Home page with upload and analysis
│   │   └── comparison.py   # Comparison page
│   └── utils/
│       ├── audio_processor.py   # Audio to spectrogram conversion
│       ├── image_processor.py   # Image preprocessing
│       ├── model_loader.py      # Model loading utilities
│       └── xai_methods.py       # XAI implementations
├── notebooks/              # Jupyter notebooks for training
├── models/                 # Saved trained models
│   ├── audio/
│   └── image/
├── data/                   # Datasets
│   ├── spectrograms/       # Audio spectrograms
│   └── LungCancerDetection/
└── requirements.txt        # Python dependencies
```

## Features in Detail

### Automatic Compatibility Filtering
The interface automatically filters XAI methods based on input type:
- Only relevant methods are shown for each data modality
- Prevents incompatible method selections
- Clear explanations of why methods are available/unavailable

### XAI Visualizations
Each XAI method provides unique insights:
- **LIME**: Highlights regions that influenced the prediction
- **SHAP**: Shows feature importance using game theory
- **Grad-CAM**: Visualizes where the model focused attention

### Results History
- All analyses are saved to session history
- Easy comparison of multiple results
- Statistical summaries and insights

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- librosa (audio processing)
- OpenCV (image processing)
- LIME, SHAP (XAI libraries)
- See `requirements.txt` for complete list

## Notes

### Demo Mode
If no trained models are found, the application will create untrained models using ImageNet pretrained weights. These models won't provide accurate classifications but demonstrate the interface functionality.

To use with actual predictions, train models using the provided notebooks first.

### GPU Support
For faster processing, especially with SHAP and Grad-CAM:
- Install TensorFlow with GPU support
- Ensure CUDA and cuDNN are properly configured

### Sample Data
Sample audio and image files should be placed in:
- Audio: `data/Deepfake-Audio-Detection-with-XAI/testing/`
- Images: `data/LungCancerDetection/processed/testing/`

## Troubleshooting

### "No models found" error
Train models using the notebooks in the `notebooks/` directory.

### LIME/SHAP import errors
Ensure all dependencies are installed:
```bash
pip install lime shap
```

### Memory issues
- Process smaller batches
- Use lower resolution images
- Reduce `num_samples` parameter in LIME

## Team

This project is developed for the CDOF course.

## License

Educational use only.

## Generative AI Usage Statement

This project was developed with assistance from Claude Code (Anthropic) for:
- Code structure and refactoring
- Documentation generation
- Debugging and optimization
- Interface design recommendations
