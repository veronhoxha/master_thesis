from __future__ import annotations

import os
from io import BytesIO
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_df_from_bytes(file_bytes: bytes, extension: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    if extension == ".csv":
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


@st.cache_data(show_spinner=False)
def load_df_from_path(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


def list_sample_files(sample_dir: str) -> list[str]:
    if not os.path.isdir(sample_dir):
        return []
    files: list[str] = []
    for name in sorted(os.listdir(sample_dir)):
        lower = name.lower()
        if lower.endswith((".csv", ".xlsx", ".xls")):
            files.append(os.path.join(sample_dir, name))
    return files


