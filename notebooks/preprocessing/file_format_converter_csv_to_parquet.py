### IMPORTS ###

import pandas as pd
import os

# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


###########################


folders = [
    ("../../data/raw/", "../../data/raw/"),
    ("../../data/preprocessed/", "../../data/preprocessed/")
]

for input_folder, output_folder in folders:
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            parquet_path = os.path.join(output_folder, filename.replace(".csv", ".parquet"))

            print(f"Converting: {filename} ({input_folder})")

            chunks = pd.read_csv(
                csv_path,
                chunksize=100_000,
                low_memory=False,
                dtype=str
            )

            # combining chunks
            df = pd.concat(chunks, ignore_index=True)

            # writing to parquet
            df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

            print(f"converted: {filename} to {os.path.basename(parquet_path)}")

print("All CSV files from both folders have been converted to Parquet.")
