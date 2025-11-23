import plotly.express as px
import streamlit as st


def render_missing_values(df) -> None:
    st.markdown("---")
    st.header("🔍 Data Quality Analysis")

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if len(missing_data) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(x=missing_data.values, y=missing_data.index, orientation='h',
                         title="Missing Values by Column", labels={'x': 'Count', 'y': 'Column'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Missing Data Summary")
            for col, count in missing_data.items():
                pct = (count / len(df)) * 100
                st.metric(col, f"{count} ({pct:.1f}%)")
    else:
        st.success("✅ No missing values detected!")


