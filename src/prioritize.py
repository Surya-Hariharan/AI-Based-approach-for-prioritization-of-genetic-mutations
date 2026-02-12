import os
# Set environment variable to allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pandas as pd
import numpy as np
import torch
import sys
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.config.data_config import Config
from src.models.baseline import LogisticRegression
from src.models.mlp import MLP
from src.ensemble.stacking import StackingEnsemble
from src.uncertainty.mc_dropout import MCDropoutEstimator
from src.ranking.bayesian import BayesianRanker
from src.aggregation.gene_score import GeneAggregator

def prioritize_mutations(input_file, config_path, model_path, ensemble_path, preprocessor_path, output_file, 
                         use_ensemble=False, use_mc_dropout=False, mc_samples=20, aggregate_genes=False, gene_col='gene_id'):
    
    # 1. Load Config
    config = Config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 3. Preprocess
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    # Separate features
    # Assuming input file has same structure as training data (minus target?)
    # We need to ensure columns match.
    # For prototype, we assume columns match numerical/categorical config
    
    # If target column exists, drop it
    if config.target_column in df.columns:
        df_features = df.drop(columns=[config.target_column])
    else:
        df_features = df
        
    # Select features
    # This logic should match data_loader.py processing
    # Simplify: apply preprocessor transform
    
    # We need to construct X exactly as data_loader does.
    # This part is tricky without refactoring data_loader to expose "transform_df" function.
    # As per previous prioritize.py implementation, we assume preprocessor handles it?
    # Previous implementation:
    # X_num = preprocessor['scaler'].transform(df[numerical])
    # X_cat = preprocessor['encoder'].transform(df[categorical])
    # X = np.hstack([X_num, X_cat.toarray()])
    
    try:
        X_num = preprocessor['scaler'].transform(df[config.numerical_features])
        X_cat = preprocessor['encoder'].transform(df[config.categorical_features])
        if hasattr(X_cat, "toarray"):
            X_cat = X_cat.toarray()
        X = np.hstack([X_num, X_cat])
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    input_dim = X.shape[1]
    
    # 4. Load Model
    model = None
    ensemble = None
    
    if use_ensemble:
        if os.path.exists(ensemble_path):
            print(f"Loading Ensemble from {ensemble_path}...")
            ensemble = StackingEnsemble.load(ensemble_path)
        else:
            print(f"Error: Ensemble not found at {ensemble_path}")
            return
    else:
        model_type = config.model['type']
        if model_type == "baseline":
            model = LogisticRegression(input_dim)
        elif model_type == "mlp":
            model = MLP(input_dim, config.model['mlp']['hidden_layers'], config.model['mlp']['dropout'])
            
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        else:
            print(f"Error: Model not found at {model_path}")
            return

    # 5. Inference
    print("Running Inference...")
    predictions = None
    variances = None
    
    if use_ensemble:
        predictions = ensemble.predict_proba(X)
        variances = None
    elif use_mc_dropout:
        print(f"Using MC Dropout ({mc_samples} samples)...")
        estimator = MCDropoutEstimator(model, n_samples=mc_samples, device=device)
        res = estimator.predict(X)
        predictions = res['mean']
        variances = res['variance']
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
            variances = None
            
    # 6. Bayesian Ranking
    print("Ranking...")
    bayesian = BayesianRanker()
    rank_res = bayesian.rank(predictions, variances)
    
    df['score'] = predictions
    df['rank'] = rank_res['ranks']
    df['posterior_mean'] = rank_res['posterior_mean']
    df['posterior_lower'] = rank_res['posterior_lower']
    
    if variances is not None:
        df['uncertainty_variance'] = variances
        
    # Sort by rank
    df_sorted = df.sort_values('rank')
    
    # 7. Gene Aggregation
    if aggregate_genes and gene_col in df.columns:
        print("Aggregating Gene Scores...")
        aggregator = GeneAggregator(method=config.aggregation.get('method', 'bayesian'))
        gene_stats = aggregator.aggregate(df_sorted, 'score', gene_col)
        
        gene_output = output_file.replace(".csv", "_genes.csv")
        gene_stats.to_csv(gene_output, index=False)
        print(f"Gene stats saved to {gene_output}")

    # Save
    df_sorted.to_csv(output_file, index=False)
    print(f"Prioritized mutations saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prioritize Mutations in New Dataset")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--ensemble_path", type=str, default="models/ensemble.joblib", help="Path to ensemble checkpoint")
    parser.add_argument("--preprocessor_path", type=str, default="models/preprocessor.joblib", help="Path to preprocessor")
    parser.add_argument("--output", type=str, default="results/prioritized.csv", help="Path to output CSV")
    
    parser.add_argument("--use_ensemble", action="store_true", help="Use Ensemble model")
    parser.add_argument("--use_mc_dropout", action="store_true", help="Use MC Dropout (single model only)")
    parser.add_argument("--mc_samples", type=int, default=20, help="MC samples")
    parser.add_argument("--aggregate_genes", action="store_true", help="Aggregate scores by gene")
    parser.add_argument("--gene_col", type=str, default="gene_id", help="Gene ID column name")
    
    args = parser.parse_args()
    
    prioritize_mutations(args.input_file, args.config, args.model_path, args.ensemble_path, args.preprocessor_path, args.output,
                         args.use_ensemble, args.use_mc_dropout, args.mc_samples, args.aggregate_genes, args.gene_col)

if __name__ == "__main__":
    main()
