import streamlit as st


def render_preview_and_download(df) -> None:
    st.markdown("---")
    st.header("🔎 Raw Data Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("First rows of dataset")
    with col2:
        num_rows = st.number_input("Rows to display", min_value=5, max_value=100, value=10, step=5)
    st.dataframe(df.head(num_rows), use_container_width=True)

    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Processed Data as CSV",
                       data=csv, file_name="processed_data.csv", mime="text/csv")


