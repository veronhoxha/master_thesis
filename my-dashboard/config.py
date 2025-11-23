import os

# Page configuration for Streamlit
PAGE_CONFIG = {
    "page_title": "Dataset Analysis Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Constants
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Default directories for datasets
ROOT_DIR = "/Users/estref/Desktop/master_thesis"
DATA_DIR = f"{ROOT_DIR}/data"
SAMPLE_DIR = f"{DATA_DIR}/preprocessed"
RAW_DIR = f"{DATA_DIR}/raw"
GENERATED_DIR = f"{DATA_DIR}/generated"

# Styles path
STYLES_PATH = os.path.join(os.path.dirname(__file__), "styles.css")


