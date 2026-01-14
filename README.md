# Unified Explainable AI Interface

A unified web application for **multi-modal AI classification** with **explainable AI (XAI) techniques**. This project integrates audio deepfake detection and lung cancer detection into a single platform with support for LIME, SHAP, and Grad-CAM visualizations.

## Project Overview

This project combines two separate AI systems:
1. **Deepfake Audio Detection**: Detects real vs. fake audio using neural networks trained on mel-spectrograms
2. **Lung Cancer Detection**: Identifies malignant tumors in chest X-rays using transfer learning

The unified interface allows users to:
- Upload audio (.wav) or image files (.jpg, .png)
- Select compatible classification models
- Apply XAI techniques (LIME, SHAP, Grad-CAM)
- Compare multiple XAI methods side-by-side
- View interactive visualizations explaining model decisions

## Tech Stack

- **Frontend**: React + Vite, React Router, Axios, CSS Modules
- **Backend**: FastAPI (Python), Pydantic for validation
- **ML Framework**: TensorFlow/Keras (for future real model integration)
- **XAI Libraries**: LIME, SHAP, OpenCV (currently using mock implementations)

## Features

### Implemented
✅ Multi-modal file upload (audio and images) with drag-and-drop
✅ Automatic model compatibility filtering based on file type
✅ Mock classification with realistic results
✅ XAI visualization generation (LIME, SHAP, Grad-CAM)
✅ Side-by-side comparison of original input and XAI explanation
✅ Responsive design (mobile, tablet, desktop)
✅ RESTful API with automatic documentation
✅ Compatibility matrix ensuring only valid XAI methods are shown

### Future Enhancements
⏳ Real model integration (load .keras files and run inference)
⏳ Audio-to-spectrogram conversion using librosa
⏳ Real XAI implementations (currently using mock visualizations)
⏳ Comparison page for side-by-side XAI method analysis
⏳ Model performance metrics and confidence intervals

## Architecture

```
Unified-Explainable-AI-Interface/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI app entry point
│   │   ├── routers/        # API endpoints
│   │   ├── utils/          # Utilities (compatibility, mock data)
│   │   └── models/         # Pydantic schemas
│   └── requirements.txt
│
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── context/        # State management
│   │   └── services/       # API communication
│   └── package.json
│
├── models/                  # Trained model weights (.keras)
├── notebooks/               # Training notebooks
└── data/                    # Datasets
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`
   - Interactive API docs: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

### Frontend Setup

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

## How to Use

### Step 1: Upload File
1. Open the application at `http://localhost:5173`
2. Drag and drop or browse to upload:
   - Audio file (.wav) for deepfake detection
   - Image file (.jpg, .png) for lung cancer detection

### Step 2: Select Model
1. After upload, compatible models will be displayed
2. Click on a model card to run classification
3. View the prediction result with confidence score

### Step 3: Choose XAI Method
1. Select an explainability method:
   - **LIME**: Local interpretable explanations with superpixel highlighting
   - **SHAP**: Shapley value-based pixel attributions
   - **Grad-CAM**: Gradient-based attention heatmaps
2. The visualization will show which regions influenced the prediction

### Step 4: View Results
1. Compare original input with XAI explanation side-by-side
2. Review metadata (processing time, explanation quality)
3. Download visualizations if needed

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and validate file |
| POST | `/api/classify` | Run classification |
| POST | `/api/xai/explain` | Generate XAI visualization |
| POST | `/api/xai/compare` | Compare multiple XAI methods |
| GET | `/api/models` | Get available models |
| GET | `/api/xai/methods` | Get XAI methods |
| GET | `/api/files/{file_id}` | Retrieve uploaded file |
| GET | `/health` | Health check |

## Models

### Audio Models (Deepfake Detection)
- **VGG16**: 83.92% accuracy - Best overall performance
- **MobileNet**: 78.22% accuracy - Most efficient
- **ResNet50**: 52.39% accuracy
- **InceptionV3**: 55.33% accuracy
- **Custom CNN**: 50.37% accuracy

### Image Models (Lung Cancer Detection)
- **DenseNet121 + VAE**: 90% accuracy (mock)
- **DenseNet121**: 88% accuracy (mock)
- **VGG16 + VAE**: 87% accuracy (mock)
- **VGG16**: 85% accuracy (mock)

## XAI Methods

| Method | Full Name | Compatible With | Description |
|--------|-----------|-----------------|-------------|
| LIME | Local Interpretable Model-agnostic Explanations | Audio, Image | Superpixel-based local explanations |
| SHAP | SHapley Additive exPlanations | Audio, Image | Game theory-based feature attributions |
| Grad-CAM | Gradient-weighted Class Activation Mapping | Audio, Image | CNN gradient-based attention maps |

## Development

### Project Structure
- **Backend**: Modular FastAPI structure with routers, services, and utilities
- **Frontend**: Component-based React architecture with CSS Modules
- **State Management**: React Context API for global state
- **API Communication**: Axios with centralized endpoint functions

### Code Quality
- Python: Type hints, Pydantic validation, async/await
- JavaScript: ES6+, functional components, hooks
- CSS: Modular styles, responsive design, mobile-first approach

### Testing
Currently, the application uses mock data for demonstration purposes. To test:

1. Start both backend and frontend servers
2. Upload a test file (audio or image)
3. Go through the full workflow: upload → classify → XAI
4. Verify visualizations display correctly
5. Check browser console for errors

## Migration to Real Models

To integrate real trained models:

1. **Update backend dependencies** in `requirements.txt`:
   ```
   tensorflow==2.18.*
   librosa>=0.9.0
   lime>=0.2.0.1
   shap>=0.41.0
   ```

2. **Implement model loading** in `backend/app/services/model_loader.py`
3. **Add audio preprocessing** in `backend/app/services/preprocessor.py`
4. **Replace mock generators** with real XAI implementations
5. **Update endpoints** to use real predictions

## Generative AI Usage Statement

This project was developed with assistance from **Claude Code (Anthropic)** for the following purposes:

- **Code Generation**: FastAPI backend structure, React components, and API integration
- **Architecture Design**: System design, component hierarchy, and state management
- **Documentation**: README, code comments, and API documentation
- **Debugging**: Error handling and code optimization

All code was reviewed and tested before inclusion in the final project.

## Contributors

- Maxime Langelier
- TD Group: [Your TD Group Here]

## License

This project is for educational purposes as part of a university assignment.

## Acknowledgments

- Original deepfake audio detection implementation
- Original lung cancer detection implementation
- TensorFlow and Keras teams
- FastAPI and React communities
- XAI library developers (LIME, SHAP)

---

**Built with ❤️ using React + FastAPI**
