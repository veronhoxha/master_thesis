"""Create ground truth column type mapping from real datasets."""

import json
import pandas as pd
from pathlib import Path

datasets = {
    'hatecrime': '../../data/preprocessed/hate_crime_preprocessed.csv',
    'employment': '../../data/preprocessed/uk_gender_pay_gap_data_2024_to_2025_preproccesed.csv',
    'lending': '../../data/preprocessed/year_2024_preprocessed.csv'
}

ground_truth = {}

for domain, path in datasets.items():
    try:
        print(f"\nAnalyzing {domain}...")
        df = pd.read_csv(path, nrows=2000) 
        print(f"  Found {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)[:10]}...")
        
        numeric = []
        categorical = []
        text = []
        
        for col in df.columns:
            dtype = df[col].dtype
            s = df[col].dropna()
            unique_count = s.nunique() if len(s) > 0 else 0
            total = len(s)
            
            if pd.api.types.is_numeric_dtype(dtype):
                numeric.append(col)
            elif total == 0:
                categorical.append(col)  
            elif unique_count <= 50 and total > 0:
                unique_fraction = unique_count / total
                if unique_fraction <= 0.5 or unique_count <= 20:
                    categorical.append(col)
                else:
                    text.append(col)
            else:
                text.append(col)
        
        ground_truth[domain] = {
            'numeric': numeric,
            'categorical': categorical,
            'text': text,
            'total_columns': len(df.columns)
        }
        
        print(f"  - Numeric: {len(numeric)} columns")
        print(f"  - Categorical: {len(categorical)} columns")
        print(f"  - Text: {len(text)} columns")
        
        if numeric:
            print(f"    Numeric examples: {numeric[:3]}")
        if categorical:
            print(f"    Categorical examples: {categorical[:3]}")
        if text:
            print(f"    Text examples: {text[:3]}")
        
    except Exception as e:
        print(f"    Error: {e}")

output_path = Path('../../data/preprocessed/column_type_ground_truth.json')
with open(output_path, 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"Ground truth saved to: {output_path}")
