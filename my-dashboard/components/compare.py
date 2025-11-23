import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_schema_compare(real_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    st.subheader("Schema Comparison")
    real_cols = set(real_df.columns)
    llm_cols = set(llm_df.columns)
    only_real = sorted(list(real_cols - llm_cols))
    only_llm = sorted(list(llm_cols - real_cols))
    both = sorted(list(real_cols & llm_cols))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Only in Real", len(only_real))
        if only_real:
            st.write(only_real)
    with col2:
        st.metric("Only in LLM", len(only_llm))
        if only_llm:
            st.write(only_llm)
    with col3:
        st.metric("Shared", len(both))
        if both:
            st.write(both)


def render_basic_stats_compare(real_df: pd.DataFrame, llm_df: pd.DataFrame, shared_numeric_cols: list[str]) -> None:
    st.subheader("Distribution Overlap (Shared Numeric Columns)")
    for col in shared_numeric_cols:
        st.markdown(f"#### {col}")
        fig = px.histogram(real_df, x=col, opacity=0.6, nbins=30, color_discrete_sequence=["#1f77b4"], title=None)
        fig2 = px.histogram(llm_df, x=col, opacity=0.6, nbins=30, color_discrete_sequence=["#ff7f0e"], title=None)
        for trace in fig2.data:
            fig.add_trace(trace)
        fig.update_layout(barmode='overlay', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def compute_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    summary = df[cols].describe().T
    summary = summary[["mean", "50%", "std", "min", "max"]]
    summary.rename(columns={"50%": "median"}, inplace=True)
    return summary


def render_summary_tables(real_df: pd.DataFrame, llm_df: pd.DataFrame, shared_numeric_cols: list[str]) -> None:
    st.subheader("Summary Statistics")
    if not shared_numeric_cols:
        st.info("No shared numeric columns to summarize.")
        return
    real_summary = compute_summary(real_df, shared_numeric_cols)
    llm_summary = compute_summary(llm_df, shared_numeric_cols)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Data**")
        st.dataframe(real_summary, use_container_width=True)
    with col2:
        st.markdown("**LLM-Generated Data**")
        st.dataframe(llm_summary, use_container_width=True)


