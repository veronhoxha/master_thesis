import streamlit as st
import numpy as np
from config import PAGE_CONFIG, STYLES_PATH, SAMPLE_DIR, GENERATED_DIR
from data_io import load_df_from_path, list_sample_files
from components.compare import render_schema_compare, render_basic_stats_compare, render_summary_tables

st.set_page_config(**PAGE_CONFIG)

try:
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
except Exception:
    pass

st.title("🔀 Compare: Real vs LLM-Generated Data")

# Select datasets: one real (from preprocessed), one LLM-generated (from generated dir)
real_files = list_sample_files(SAMPLE_DIR)
llm_files = list_sample_files(GENERATED_DIR)

col1, col2 = st.columns(2)
with col1:
    real_path = st.selectbox("Select Real Dataset", options=["— None —"] + real_files, index=0)
with col2:
    llm_path = st.selectbox("Select LLM-Generated Dataset", options=["— None —"] + llm_files, index=0)

if real_path == "— None —" or llm_path == "— None —":
    st.info("Select both a real dataset and an LLM-generated dataset to compare.")
    st.stop()

real_df = load_df_from_path(real_path)
llm_df = load_df_from_path(llm_path)

render_schema_compare(real_df, llm_df)

shared_numeric = list(set(real_df.select_dtypes(include=[np.number]).columns)
                      & set(llm_df.select_dtypes(include=[np.number]).columns))

render_basic_stats_compare(real_df, llm_df, sorted(shared_numeric))
render_summary_tables(real_df, llm_df, sorted(shared_numeric))


