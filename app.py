import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from pages import home, comparison

# Page configuration
st.set_page_config(
    page_title="Unified XAI Interface",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Comparison"]
)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

if 'current_file' not in st.session_state:
    st.session_state.current_file = None

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Route to appropriate page
if page == "Home":
    home.show()
elif page == "Comparison":
    comparison.show()
