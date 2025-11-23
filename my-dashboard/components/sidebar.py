import streamlit as st
from typing import Dict, List, Optional, Tuple


def sidebar_controls(sample_files: List[str]) -> Tuple[Optional[object], Optional[str], Dict]:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file"
    )

    st.markdown("---")

    selected_sample = None
    if sample_files:
        st.subheader("Or select a sample dataset")
        sample_choice = st.selectbox("Sample datasets", options=["— None —"] + sample_files, index=0)
        if sample_choice != "— None —":
            selected_sample = sample_choice
            st.success("✅ Sample selected")

    has_data_selection = uploaded_file is not None or selected_sample is not None
    options = {
        "show_nulls": True,
        "show_correlations": True,
        "show_distributions": True,
    }
    if has_data_selection:
        st.subheader("Analysis Options")
        options["show_nulls"] = st.checkbox("Show null value analysis", value=True)
        options["show_correlations"] = st.checkbox("Show correlation matrix", value=True)
        options["show_distributions"] = st.checkbox("Show distributions", value=True)

    return uploaded_file, selected_sample, options


