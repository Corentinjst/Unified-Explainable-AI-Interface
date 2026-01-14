# Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then install all required packages:

```bash
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install streamlit tensorflow librosa lime shap opencv-python numpy pandas pillow matplotlib seaborn scikit-learn
```

### 2. Run the Application

**On macOS/Linux:**
```bash
./run_app.sh
```

Or directly:
```bash
streamlit run app.py
```

**On Windows:**
```bash
run_app.bat
```

Or directly:
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## First Time Usage

### Without Trained Models (Demo Mode)

If you haven't trained any models yet, the app will work in demo mode:
- Upload any .wav audio file or .jpg/.png image
- Select a model architecture
- The app will use pretrained ImageNet weights for inference
- Results won't be accurate but demonstrate the interface

### With Trained Models

To use the app with real predictions:

1. **Train Audio Models** (for deepfake detection):
   ```bash
   # First convert audio to spectrograms
   jupyter notebook notebooks/00_audio_to_spectrogram.ipynb

   # Train models
   jupyter notebook notebooks/01_train_vgg16.ipynb
   ```

   Trained models will be saved to `models/audio/`

2. **Train Image Models** (for lung cancer detection):
   ```bash
   # Process the dataset
   jupyter notebook notebooks/06_processing_lung_cancer.ipynb

   # Train models
   jupyter notebook notebooks/09_train_densenet_without_vae.ipynb
   ```

   Trained models will be saved to `models/image/`

## Quick Usage Tutorial

### Step 1: Upload a File
- Go to the Home page
- Drag & drop or browse for a file:
  - **Audio**: .wav files (for deepfake detection)
  - **Image**: .jpg, .png, .jpeg (for chest X-ray analysis)

### Step 2: Select a Model
Choose from available models:
- **Audio Models**: VGG16, MobileNet, ResNet50, InceptionV3
- **Image Models**: AlexNet, DenseNet, VGG16

### Step 3: Choose XAI Method
Select an explainability technique:
- **LIME**: Highlights influential regions
- **SHAP**: Shows feature importance
- **Grad-CAM**: Visualizes model attention

### Step 4: Analyze
Click "Run Classification & Explanation" to see:
- Prediction result with confidence score
- XAI visualization explaining the decision
- Detailed interpretation guide

### Step 5: Compare Results (Optional)
- Navigate to the "Comparison" page
- Select multiple previous results
- View side-by-side comparisons
- Analyze differences between models/methods

## Troubleshooting

### Port Already in Use
If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
If you encounter memory errors:
- Close other applications
- Process smaller files
- Use simpler models (e.g., MobileNet instead of ResNet50)

### Model Not Found
If you see "model not found" errors:
- The app will create untrained models automatically
- For real predictions, train models using the notebooks

## Example Workflow

1. **Audio Analysis Example**:
   ```
   Upload: sample_audio.wav
   Model: VGG16
   XAI: Grad-CAM
   Result: "FAKE AUDIO (Confidence: 87.3%)"
   ```

2. **Image Analysis Example**:
   ```
   Upload: chest_xray.jpg
   Model: DenseNet
   XAI: LIME
   Result: "MALIGNANT (Confidence: 92.1%)"
   ```

3. **Compare Results**:
   - Analyze the same audio with different models
   - Compare LIME vs SHAP vs Grad-CAM explanations
   - View statistical summaries

## Tips

- **XAI Method Selection**:
  - LIME: Best for understanding specific regions
  - SHAP: Best for overall feature importance
  - Grad-CAM: Best for CNN models, shows attention

- **Performance**:
  - First analysis may be slower (model loading)
  - Subsequent analyses are cached and faster
  - GPU support significantly speeds up SHAP

- **Best Practices**:
  - Use the comparison page to validate predictions
  - Try multiple XAI methods for comprehensive understanding
  - Compare different models to ensure consistency

## Need Help?

Check the main README.md for:
- Detailed project structure
- Complete feature list
- Training instructions
- Architecture details

## Ready to Start?

Run the app:
```bash
streamlit run app.py
```

Happy analyzing! 🔬
