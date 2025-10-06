import pandas as pd
import os
import json

folders = [
    ("../../data/raw/", "../../data/raw/"),
    ("../../data/preprocessed/", "../../data/preprocessed/")
]

for input_folder, output_folder in folders:
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            json_path = os.path.join(output_folder, filename.replace(".csv", ".json"))

            print(f"Converting: {filename} ({input_folder})")

            chunksize = 100_000
            first_record = True

            with open(json_path, "w", encoding="utf-8") as json_file:
                json_file.write("[\n")

                for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str, low_memory=False):
                    for _, row in chunk.iterrows():
                        if not first_record:
                            json_file.write(",\n")
                        else:
                            first_record = False
                        json.dump(row.to_dict(), json_file, ensure_ascii=False)

                json_file.write("\n]")

            print(f"Converted: {filename} to {os.path.basename(json_path)}")

print("All CSV files from both folders have been converted to JSON.")
