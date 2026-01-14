@echo off
REM Startup script for the Unified XAI Interface (Windows)

echo Starting Unified Explainable AI Interface...
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Error: Streamlit is not installed.
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import streamlit; import tensorflow; import librosa; import lime; import shap; import cv2; print('All required packages are installed')"
if errorlevel 1 (
    echo.
    echo Error: Some required packages are missing.
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Launching Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
