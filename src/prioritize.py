import argparse
import torch
import pandas as pd
import numpy as np
import os
import sys
import joblib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config.data_config import Config
from src.models.baseline import LogisticRegression
from src.models.mlp import MLP
from src.evaluation.ranker import Ranker
from src.utils.preprocessing import Preprocessor

def main():
    parser = argparse.ArgumentParser(description="Prioritize Mutations in New Data")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth", help="Path to trained model")
    parser.add_argument("--preprocessor_path", type=str, default="models/preprocessor.joblib", help="Path to saved preprocessor")
    parser.add_argument("--output", type=str, default="prioritized_mutations.csv", help="Output filename")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    # Note: We need input_dim. For inference, we determine it from preprocessor/data.
    # Alternatively, store input_dim in config or model metadata.
    # Here we'll load the model structure based on config, but input_dim is tricky.
    # Let's verify standard dimensionality from config features + OHE?
    # Safer: Load preprocessor first, transform dummy row to get dim.
    
    # Load Preprocessor
    if not os.path.exists(args.preprocessor_path):
        print(f"Error: Preprocessor not found at {args.preprocessor_path}")
        return
    
    # We use joblib directly as Preprocessor class wrapper might need instantiation with args first
    # Or we can use the load method if we instantiate.
    # Let's instantiate wrapper.
    preprocessor = Preprocessor(config.numerical_features, config.categorical_features)
    preprocessor.load(args.preprocessor_path)
    
    # Load Data
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
        
    # Preprocess
    try:
        X_processed = preprocessor.transform(df)
        input_dim = X_processed.shape[1]
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
        
    X_tensor = torch.tensor(X_processed.values, dtype=torch.float32).to(device)

    # Load Model structure
    model_type = config.model['type']
    if model_type == "baseline":
        model = LogisticRegression(input_dim)
    elif model_type == "mlp":
        model = MLP(input_dim, config.model['mlp']['hidden_layers'], config.model['mlp']['dropout'])
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
    # Rank
    ranker = Ranker(output_dir=os.path.dirname(args.output) or ".")
    
    # Check for Variant ID column
    variant_ids = None
    if "variant_id" in df.columns:
        variant_ids = df['variant_id'].tolist()
        
    ranked_df = ranker.rank_mutations(probs, df, variant_ids=variant_ids)
    
    # Save
    ranked_df.to_csv(args.output, index=False)
    print(f"Prioritized mutations saved to {args.output}")

if __name__ == "__main__":
    main()
