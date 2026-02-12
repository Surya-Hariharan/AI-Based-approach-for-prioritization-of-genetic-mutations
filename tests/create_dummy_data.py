import pandas as pd
import numpy as np
import os

def create_dummy_data(output_path="data/processed/feature_matrix.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_samples = 100
    data = {
        "AF": np.random.rand(n_samples),
        "CADD_PHRED": np.random.rand(n_samples) * 30,
        "conservation_score": np.random.rand(n_samples),
        "variant_type": np.random.choice(["SNV", "indel"], n_samples),
        "impact_category": np.random.choice(["HIGH", "MODERATE", "LOW"], n_samples),
        "pathogenicity_label": np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dummy data created at {output_path}")

if __name__ == "__main__":
    create_dummy_data("tests/data/feature_matrix.csv")
