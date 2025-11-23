import streamlit as st


def render_header() -> None:
    st.info("👈 Please upload a dataset using the sidebar to begin analysis")
    st.markdown("### Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📈 **Statistical Metrics**")
        st.markdown("- Mean, Median, Mode")
        st.markdown("- Standard Deviation")
        st.markdown("- Min/Max Values")
    with col2:
        st.markdown("📊 **Visualizations**")
        st.markdown("- Distribution plots")
        st.markdown("- Correlation heatmaps")
        st.markdown("- Time series analysis")
    with col3:
        st.markdown("🔍 **Data Quality**")
        st.markdown("- Missing values")
        st.markdown("- Data types")
        st.markdown("- Dataset overview")


