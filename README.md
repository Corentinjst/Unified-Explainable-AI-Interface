# Unified Explainable AI Interface

A unified web application for **multi-modal AI classification** with **explainable AI (XAI) techniques**. This project integrates deepfake audio detection and lung cancer detection into a single platform with full support for LIME, SHAP, and Grad-CAM visualizations.

## Team Information

**Team Members:**
- Corentin JUSTE
- Maxime Langelier
- Noé Le YHUELIC
- Vianney LE BHOURIS

**TD Group:** DIA 4

**Course:** Explainability AI


## Project Overview

This project successfully integrates two existing Explainable AI systems into a single interactive platform:

1. **Deepfake Audio Detection**: Detects real vs. fake audio using neural networks (VGG16, MobileNet, InceptionV3, Custom CNN) trained on mel-spectrograms from the Fake-or-Real (FoR) dataset
2. **Lung Cancer Detection**: Identifies malignant tumors in chest X-rays using transfer learning models (DenseNet121, VGG16) on the CheXpert dataset


The unified interface allows users to:
- Upload audio (.wav) or image files
- Select compatible classification models automatically filtered by input type
- Apply three XAI techniques (LIME, SHAP, Grad-CAM) with real implementations
- Compare multiple XAI methods side-by-side on the same input
- View interactive visualizations with metadata explaining model decisions


## Tech Stack

### Frontend
- **Framework**: React 18 + Vite
- **Styling**: Vanilla CSS

### Backend
- **Framework**: FastAPI 

### Machine Learning
- **Framework**: TensorFlow 2.18 / Keras
- **XAI Libraries**:
  - LIME
  - SHAP
  - scikit-image (for Grad-CAM)




## Basic Workflow (Analysis Page)
Complete step-by-step workflow:
1. **Upload**: Drag-and-drop or file browser
2. **Model Selection**: Choose from compatible models
3. **Classification**: View prediction with confidence scores
4. **XAI Selection**: Choose explainability method
5. **Visualization**: Side-by-side comparison of original input and explanation



## Setup Instructions

### Backend

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**:
   - Place `.keras` model files in `models/audio/` and `models/image/` directories
   - Audio models: `vgg16_model.keras`, `mobilenet_model.keras`, etc.
   - Image models: `densenet121_model.keras`, `vgg16_model.keras`, etc.

5. **Run the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:5173`



## Models

### Audio Models (Deepfake Detection)

Trained on the Fake-or-Real (FoR) dataset using mel-spectrograms:

| Model | Architecture | Accuracy | Parameters |
|-------|-------------|----------|------------|
| VGG16 | Transfer Learning | 83.92% | ~14.7M | 
| MobileNet | Transfer Learning | 78.22% | ~3.2M |
| InceptionV3 | Transfer Learning | 55.33% | ~21.8M | 
| Custom CNN | From Scratch | 50.37% | ~500K | 

**Input Format**: Audio → Mel-spectrogram (224×224, 128 mel-bands)
**Classes**: `["fake", "real"]`

### Image Models (Lung Cancer Detection)

Trained on the CheXpert dataset:

| Model | Architecture | Accuracy | Parameters | 
|-------|-------------|----------|------------|
| DenseNet121 | Transfer Learning | 88% | ~7M | 
| VGG16 | Transfer Learning | 85% | ~14.7M | 

**Input Format**: Chest X-ray images (224×224)
**Classes**: `["benign", "malignant"]`



## Generative AI Usage Statement

This project was developed with assistance from **Anthropic's Claude Code** for the following purposes:

- All Frontend Work (except minor CSS modifications)
- Figuring out the file structure of the project (namely the split into multiple XAI services and BaseExplainer class)
- Fixing the vanilla GradCAM explainer implementation that had tensorflow-related issues.
- Thourougly evaluate the code that was human-written to ensure good code quality.


All AI-generated code was reviewed, tested, and modified as needed to ensure correctness and alignment with project requirements. The use of Generative AI was primarily for accelerating development and ensuring code quality, not for bypassing learning objectives.



**Original Sources:**
- [Deepfake Audio Detector with XAI](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI)
- [Lung Cancer Detection](https://github.com/schaudhuri16/LungCancerDetection)
