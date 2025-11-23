import streamlit as st


def render_stat_summaries(df, selected_numeric) -> None:
    st.header("📈 Statistical Summary")
    for col in selected_numeric:
        with st.expander(f"📊 {col}", expanded=True):
            metric_cols = st.columns(6)
            with metric_cols[0]:
                st.metric("Mean", f"{df[col].mean():.2f}")
            with metric_cols[1]:
                st.metric("Median", f"{df[col].median():.2f}")
            with metric_cols[2]:
                st.metric("Std Dev", f"{df[col].std():.2f}")
            with metric_cols[3]:
                st.metric("Min", f"{df[col].min():.2f}")
            with metric_cols[4]:
                st.metric("Max", f"{df[col].max():.2f}")
            with metric_cols[5]:
                st.metric("Missing", f"{df[col].isnull().sum()}")


