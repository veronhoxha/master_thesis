import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import io
import zipfile
import tempfile

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

st.set_page_config(page_title="Generate with LLM", page_icon="🧪", layout="wide")

st.title("🧪 Generate with LLM")
st.caption("Replicates the batch generation from the notebook using HF + Kimi, writes CSVs locally, and lets you download them.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASH_OUT_DIR = PROJECT_ROOT / "my-dashboard" / "generated"
DASH_OUT_DIR.mkdir(parents=True, exist_ok=True)

with st.form("batch_form"):
    st.subheader("Batch Parameters")
    models = st.multiselect("Models", ["moonshotai/Kimi-K2-Instruct-0905"], default=["moonshotai/Kimi-K2-Instruct-0905"]) 
    domains = st.multiselect("Domains", ["hatecrime", "employment", "lending"], default=["hatecrime", "employment", "lending"]) 
    shots = st.multiselect("Shot Types", ["zero", "one", "few"], default=["zero", "one", "few"]) 
    run_tag = st.selectbox("Run Tag", ["run1", "run2", "run3"], index=0)
    rows_per_chunk = st.number_input("Rows per chunk", 5, 200, 20, 5)
    chunks = st.number_input("Chunks per experiment", 1, 100, 5, 1)

    submitted = st.form_submit_button("Run Batch Generation")

if submitted:
    token = os.getenv("MY_ENV_VAR") or os.getenv("HF_TOKEN")
    if not token:
        st.error("Hugging Face token not found. Set MY_ENV_VAR or HF_TOKEN.")
        st.stop()
    if InferenceClient is None:
        st.error("huggingface_hub not installed in environment.")
        st.stop()

    # simple domain schemas
    schemas = {
        "employment": [
            "EmployerName", "SicCodes", "DiffMeanHourlyPercent", "DiffMedianHourlyPercent",
            "DiffMeanBonusPercent", "DiffMedianBonusPercent", "MaleBonusPercent", "FemaleBonusPercent",
            "MaleLowerQuartile", "FemaleLowerQuartile", "MaleLowerMiddleQuartile", "FemaleLowerMiddleQuartile",
            "MaleUpperMiddleQuartile", "FemaleUpperMiddleQuartile", "MaleTopQuartile", "FemaleTopQuartile",
            "EmployerSize", "DateSubmitted"
        ],
        "lending": [
            "applicant_sex", "derived_sex", "derived_ethnicity", "derived_race", "applicant_age",
            "loan_amount", "rate_spread", "action_taken", "loan_originated"
        ],
        "hatecrime": [
            "offender_race", "offender_ethnicity", "victim_types", "bias_desc", "offense_count", "arrest_made"
        ],
    }

    def prompt_for(headers: list, n: int) -> str:
        cols = ",".join(headers)
        return (
            "You are a data generator. Output ONLY valid CSV with the exact header and exactly "
            f"{n} data rows. No commentary.\n"
            f"Header: {cols}\n"
            "Rules: Use plausible values per column name, avoid quotes inside fields; use ISO dates where relevant."
        )

    total_jobs = len(models) * len(domains) * len(shots)
    job_idx = 0
    log = []

    for model in models:
        client = InferenceClient(model=model, token=token)
        model_slug = model.split("/")[-1].lower().replace("_", "-")
        for domain in domains:
            headers = schemas.get(domain, ["col1","col2","col3"]) 
            for shot in shots:
                job_idx += 1
                st.info(f"[{job_idx}/{total_jobs}] {model} | {domain} | {shot}")

                out_dir = DASH_OUT_DIR / f"{model_slug}-{run_tag}" / domain / shot / "csv"
                out_dir.mkdir(parents=True, exist_ok=True)

                combined = []
                for ci in range(1, int(chunks) + 1):
                    p = prompt_for(headers, int(rows_per_chunk))
                    try:
                        text = client.text_generation(p, max_new_tokens=int(rows_per_chunk * 20), temperature=0.7, do_sample=True)
                    except Exception as e:
                        text = ",".join(headers) + "\n"
                        log.append((model, domain, shot, ci, f"error: {e}"))

                    if not text.strip().splitlines()[0].strip().replace(" ","") == ",".join(headers).replace(" ",""):
                        text = ",".join(headers) + "\n" + "\n".join([l for l in text.splitlines() if "," in l])

                    try:
                        df = pd.read_csv(io.StringIO(text))
                    except Exception:
                        lines = [l for l in text.splitlines() if "," in l]
                        df = pd.read_csv(io.StringIO(",".join(headers) + "\n" + "\n".join(lines)), on_bad_lines='skip')

                    if len(df) > rows_per_chunk:
                        df = df.iloc[: rows_per_chunk].copy()
                    elif len(df) < rows_per_chunk:
                        df = pd.concat([df, pd.DataFrame([{}] * (rows_per_chunk - len(df)))], ignore_index=True)
                        df = df.reindex(columns=headers)

                    df.to_csv(out_dir / f"chunk_{ci:03d}_attempt_1_parsed.csv", index=False)
                    combined.append(df)

                if combined:
                    final_df = pd.concat(combined, ignore_index=True)
                    final_name = f"{model_slug}-{run_tag}-raw_finals_of_all_chunks_{domain}_{shot}_csv.csv"
                    final_df.to_csv(out_dir / final_name, index=False)

    # Create a zip with all generated outputs for download
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in DASH_OUT_DIR.rglob("*.csv"):
            zf.write(p, p.relative_to(DASH_OUT_DIR))
    st.success("Batch generation finished.")
    st.download_button("Download generated CSVs (zip)", data=open(zip_path, 'rb').read(), file_name=f"generated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip")

    if log:
        st.subheader("Generation notes")
        df_log = pd.DataFrame(log, columns=["model","domain","shot","chunk","note"])
        st.dataframe(df_log, use_container_width=True)
