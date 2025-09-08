import streamlit as st
import pandas as pd

st.set_page_config(page_title="LLM Dataset Analysis", layout="wide")

st.title("LLM Dataset Analysis")
tabs = st.tabs(["Generate", "Audit", "Compare", "Reports"])

with tabs[0]:
    st.subheader("Generate Dataset")

    prompt = st.text_area("Enter prompt", "Generate employee dataset...")
    rows = st.number_input("Rows", 100, 10000, 1000)
    columns = st.number_input("Columns", 5, 50, 10)
    model = st.selectbox("Model", ["GPT-4o-mini", "Claude", "LLaMA"])
    fmt = st.selectbox("Format", ["CSV", "JSON", "Parquet"])

    if "gen_done" not in st.session_state:
        st.session_state.gen_done = False
    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    if st.button("Run Generate"):
        st.session_state.gen_done = True
        st.session_state.show_report = False 
    
    if st.session_state.gen_done:
        st.success("Dataset generated")

        if st.button("Show Report"):
            st.session_state.show_report = True

        st.download_button("Download Dataset", "Dataset", file_name="dataset.csv")
        st.download_button("Download Report", "Report content here", file_name="report.html")

    if st.session_state.show_report:
        st.info("Report")
        st.metric("Model", model)
        st.metric("Rows", rows)
        st.metric("Columns", columns)
        st.metric("Duplicates", "2")
        st.metric("Null Values", "20")
        st.metric("Bias", "Gender RR=0.83")

        
    
with tabs[1]:
    st.subheader("Audit Dataset")
    st.file_uploader("Upload dataset", type=["csv","json","parquet"])
    if st.button("Run Audit"):
        st.info("Audit results")
        st.metric("Rows", "1000")
        st.metric("Columns", "10")
        st.metric("Duplicates", "2")
        st.metric("Null Values", "20")
        st.metric("Bias", "Gender RR=0.83")

with tabs[2]:
    st.subheader("Compare Datasets")
    st.file_uploader("Upload real dataset")
    st.file_uploader("Upload generated dataset")
    if st.button("Run Compare"):
        st.info("Comparison results")

with tabs[3]:
    st.subheader("Reports")
    st.write("Download previous reports here.")