"""
College Chatbot - Student Interface
Wraps the unified Voice+Text interface
"""

import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="College Information Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit elements to make it look like a standalone app
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        iframe {
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE = "http://localhost:8000"

# Embed the unified Voice+Text interface
try:
    # Use full height of the viewport
    components.iframe(f"{API_BASE}/voice", height=1000, scrolling=True)
except Exception as e:
    st.error(f"Could not load interface. Ensure backend is running at {API_BASE}")
    st.error(str(e))