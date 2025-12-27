### IMPORTS ###

import pandas as pd
import numpy as np
import os
import json


# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


###########################


def convert_csv_to_json(input_folder, output_folder, chunksize=100_000):
    """Convert CSV files to JSON format with proper handling of all data types."""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            json_path = os.path.join(output_folder, filename.replace(".csv", ".json"))
            
            print(f"Converting: {filename}")
            
            try:
                first_chunk = True
                
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json_file.write("[\n")
                    
                    first_record = True
                    
                    # reading CSV in chunks for memory efficiency
                    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
                        # converting chunk to records
                        records = chunk.to_dict(orient='records')
                        
                        for record in records:
                            cleaned_record = clean_record(record)
                            
                            if not first_record:
                                json_file.write(",\n")
                            else:
                                first_record = False
                            
                            json.dump(cleaned_record, json_file, ensure_ascii=False, default=str)
                    
                    json_file.write("\n]")
                
                print(f"Converted: {filename} → {os.path.basename(json_path)}")
                
            except Exception as e:
                print(f"✗ Error converting {filename}: {str(e)}")

def clean_record(record):
    """Clean a single record by handling NaN, infinity, and other special values."""
    cleaned = {}
    
    for key, value in record.items():
        # handle pandas NA, NaN, None
        if pd.isna(value):
            cleaned[key] = None
        # handle numpy types
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            cleaned[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            if np.isinf(value):
                cleaned[key] = None 
            else:
                cleaned[key] = float(value)
        elif isinstance(value, np.bool_):
            cleaned[key] = bool(value)
        elif isinstance(value, (np.ndarray, list)):
            cleaned[key] = [clean_value(v) for v in value]
        elif pd.api.types.is_datetime64_any_dtype(type(value)):
            cleaned[key] = str(value)
        else:
            cleaned[key] = value
    
    return cleaned

def clean_value(value):
    """Helper function to clean individual values in arrays."""
    if pd.isna(value):
        return None
    elif isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        if np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    else:
        return value

# main execution
if __name__ == "__main__":
    folders = [
        ("../../data/raw/", "../../data/raw/"),
        ("../../data/preprocessed/", "../../data/preprocessed/")
    ]
    
    for input_folder, output_folder in folders:
        if os.path.exists(input_folder):
            print(f"\nProcessing folder: {input_folder}")
            convert_csv_to_json(input_folder, output_folder)
        else:
            print(f"Folder not found: {input_folder}")
    
    print("\nAll CSV files have been converted to JSON.")