import pandas as pd


def sample_large_csv(path, sample_size=200, out_path="sample.csv", chunksize=100000):
    sample_list = []

    # reading the dataset in chunks
    for chunk in pd.read_csv(path, chunksize=chunksize):
        sample_list.append(chunk.sample(n=min(sample_size, len(chunk))))

    # combine samples from all chunks
    sample_df = pd.concat(sample_list).sample(n=sample_size)
    sample_df.to_csv(out_path, index=False)
    print(f"Sample saved: {out_path}")
    
    
datasets_raw = {
    "hate_crime": "../../data/raw/hate_crime.csv",
    "gender_pay_gap": "../../data/raw/uk_gender_pay_gap_data_2024_to_2025.csv",
    "lending": "../../data/raw/year_2024.csv"  # 4GB+, using chunk sampling
}

datasets_pre = {
    "hate_crime": "../../data/preprocessed/hate_crime_preprocessed.csv",
    "gender_pay_gap": "../../data/preprocessed/uk_gender_pay_gap_data_2024_to_2025_preproccesed.csv",
    "lending": "../../data/preprocessed/year_2024_preprocessed.csv"
}


def make_samples(datasets, folder="raw"):
    for name, path in datasets.items():
        out_path = f"../../data/{folder}/sample_{name}.csv"
        try:
            if name == "lending":
                sample_large_csv(path, sample_size=200, out_path=out_path)
            else:
                df = pd.read_csv(path)
                sample = df.sample(n=200)
                sample.to_csv(out_path, index=False)
                print(f"Sample saved: {out_path}")
        except Exception as e:
            print(f"Could not process {name} in {folder}: {e}")
            
            
make_samples(datasets_raw, folder="raw")
make_samples(datasets_pre, folder="preprocessed")