#!/bin/bash
# Startup script for the Unified XAI Interface

echo "Starting Unified Explainable AI Interface..."
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "
import sys
try:
    import streamlit
    import tensorflow
    import librosa
    import lime
    import shap
    import cv2
    print('✓ All required packages are installed')
except ImportError as e:
    print(f'Error: Missing package - {e}')
    print('Please install requirements: pip install -r requirements.txt')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "Launching Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
