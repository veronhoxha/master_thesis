import numpy as np
import streamlit as st


def render_overview_metrics(df) -> None:
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    with col5:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing Data", f"{missing_pct:.1f}%")


