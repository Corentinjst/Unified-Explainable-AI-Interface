# Quick Start Guide

## Running the Application

### Terminal 1 - Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will run at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Terminal 2 - Frontend (React)

```bash
cd frontend
npm install  # Only needed first time
npm run dev
```

Frontend will run at: `http://localhost:5173`

## Test the Application

1. Open `http://localhost:5173` in your browser
2. Upload a test file:
   - For audio: Use any .wav file
   - For images: Use any .jpg or .png file
3. Select a model from the compatible models shown
4. Wait for classification result
5. Select an XAI method (LIME, SHAP, or Grad-CAM)
6. View the side-by-side visualization

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
uvicorn app.main:app --reload --port 8001
```
Update frontend API URL in `frontend/src/services/api.js` to `http://localhost:8001`

**Import errors:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend Issues

**Port 5173 already in use:**
The app will automatically try port 5174, 5175, etc.

**React dependencies error:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
Ensure backend is running first, then start frontend.

## Project Status

✅ **Completed:**
- Full backend API with mock data
- Complete React frontend with all components
- File upload (drag-drop)
- Model selection with compatibility filtering
- Mock classification
- Mock XAI visualizations (LIME, SHAP, Grad-CAM)
- Responsive design

⏳ **Future Work:**
- Integrate real TensorFlow models
- Implement real XAI algorithms
- Add comparison page for multi-XAI analysis
- Audio to spectrogram conversion

## Demo Workflow

1. **Upload**: Drag and drop test.wav or test.jpg
2. **Select Model**: Click on VGG16 (highest accuracy)
3. **View Classification**: See predicted class and confidence
4. **Select XAI**: Choose LIME for interpretable explanations
5. **Analyze**: Compare original vs. explanation visualization

Enjoy exploring explainable AI!
