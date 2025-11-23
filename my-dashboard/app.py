import streamlit as st
from config import PAGE_CONFIG, STYLES_PATH

st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
try:
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
except Exception:
    pass

# Title
st.title("📊 Dataset Analysis Dashboard")
st.markdown("Welcome! Use the Pages menu (left) to access:")
st.markdown("- 📄 Data Analysis")
st.markdown("- 🔀 Compare: Real vs LLM-Generated Data")
st.markdown("- 🧪 Generate Data with LLM")

st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Choose a page to get started")