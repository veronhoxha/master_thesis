import streamlit as st
import numpy as np
from config import PAGE_CONFIG, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, SAMPLE_DIR, STYLES_PATH
from data_io import load_df_from_bytes, load_df_from_path, list_sample_files
from validation import validate_uploaded_file
from components.sidebar import sidebar_controls
from components.header import render_header
from components.metrics import render_overview_metrics
from components.summaries import render_stat_summaries
from components.charts import render_distributions, render_boxplots, render_corr, render_scatter
from components.data_quality import render_missing_values
from components.table import render_preview_and_download

st.set_page_config(**PAGE_CONFIG)

try:
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
except Exception:
    pass

st.title("📄 Data Analysis")

with st.sidebar:
    uploaded_file, selected_sample, options = sidebar_controls(
        sample_files=list_sample_files(SAMPLE_DIR)
    )

if uploaded_file is None and selected_sample is None:
    render_header()
    st.stop()

try:
    if uploaded_file is not None:
        ext = validate_uploaded_file(uploaded_file, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS)
        df = load_df_from_bytes(uploaded_file.getvalue(), ext)
    else:
        df = load_df_from_path(selected_sample)

    render_overview_metrics(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("📊 Select Columns for Analysis")
    selected_numeric = st.multiselect(
        "Numeric Columns", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))]
    )

    if selected_numeric:
        render_stat_summaries(df, selected_numeric)
        st.header("📊 Visualizations")
        if options.get("show_distributions") and selected_numeric:
            render_distributions(df, selected_numeric)
        if len(selected_numeric) > 0:
            render_boxplots(df, selected_numeric)
        if options.get("show_correlations") and len(selected_numeric) > 1:
            render_corr(df, selected_numeric)
        if len(selected_numeric) >= 2:
            render_scatter(df, selected_numeric)

    if options.get("show_nulls"):
        render_missing_values(df)

    render_preview_and_download(df)
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.info("Please make sure your file is a valid CSV or Excel file.")


