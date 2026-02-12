import argparse
import torch
import pandas as pd
import numpy as np
import joblib

from src.config.data_config import Config
from src.utils.data_loader import get_data_loaders
from src.models.baseline import LogisticRegression
from src.models.mlp import MLP
from src.evaluation.eval_metrics import calculate_metrics, calculate_top_k_recall, confusion_matrix_stats, brier_score, expected_calibration_error
from src.evaluation.eval_plotting import Plotter
from src.evaluation.eval_report import EvaluationReport
from src.evaluation.ranker import Ranker
from src.evaluation.biological import BiologicalEvaluator
from src.interpretation.explainer import ModelInterpreter
from src.ensemble.stacking import StackingEnsemble
from src.uncertainty.mc_dropout import MCDropoutEstimator
from src.ranking.bayesian import BayesianRanker
from src.aggregation.gene_score import GeneAggregator

def main():
    parser = argparse.ArgumentParser(description="Evaluate Genetic Mutation Prioritization Model")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth", help="Path to trained model")
    parser.add_argument("--ensemble_path", type=str, default="models/ensemble.joblib", help="Path to trained ensemble")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble model")
    parser.add_argument("--use_mc_dropout", action="store_true", help="Enable MC Dropout for uncertainty")
    parser.add_argument("--mc_samples", type=int, default=20, help="Number of MC samples")
    parser.add_argument("--aggregate_genes", action="store_true", help="Perform gene-level aggregation")
    parser.add_argument("--gene_col", type=str, default="gene_id", help="Column name for gene ID")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data (Test Set)
    _, _, test_loader, input_dim = get_data_loaders(config)
    
    # Load Model or Ensemble
    model = None
    ensemble = None
    
    if args.use_ensemble:
        if os.path.exists(args.ensemble_path):
            print(f"Loading Ensemble from {args.ensemble_path}...")
            ensemble = StackingEnsemble.load(args.ensemble_path)
        else:
            print(f"Error: Ensemble not found at {args.ensemble_path}")
            return
    else:
        # Load Single Model
        model_type = config.model['type']
        if model_type == "baseline":
            model = LogisticRegression(input_dim)
        elif model_type == "mlp":
            model = MLP(input_dim, config.model['mlp']['hidden_layers'], config.model['mlp']['dropout'])
        
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            model.eval()
        else:
            print(f"Error: Model not found at {args.model_path}")
            return
    
    # Run Inference
    y_true = []
    y_scores = []
    y_variances = [] # For uncertainty
    X_all = []
    
    print("Running Inference...")
    
    # We need to iterate differently depending on model type
    if args.use_ensemble:
        # Ensemble expects full X array, not batches usually, but we can iterate batches
        for X, y in test_loader:
             # Ensemble predict_proba expects numpy
             X_np = X.numpy()
             probs = ensemble.predict_proba(X_np)
             
             y_true.extend(y.numpy().flatten())
             y_scores.extend(probs)
             X_all.extend(X.numpy())
             y_variances.extend(np.zeros_like(probs)) # No variance for ensemble yet
             
    elif args.use_mc_dropout:
        print(f"Using MC Dropout with {args.mc_samples} samples...")
        estimator = MCDropoutEstimator(model, n_samples=args.mc_samples, device=device)
        for X, y in test_loader:
            res = estimator.predict(X)
            y_true.extend(y.numpy().flatten())
            y_scores.extend(res['mean']) # Use mean as prediction
            y_variances.extend(res['variance'])
            X_all.extend(X.numpy())
            
    else:
        # Standard Inference
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.sigmoid(outputs)
                
                y_true.extend(y.cpu().numpy().flatten())
                y_scores.extend(probs.cpu().numpy().flatten())
                X_all.extend(X.cpu().numpy())
                y_variances.extend(np.zeros(len(y)))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_variances = np.array(y_variances)
    X_all = np.array(X_all)
    
    # 1. Metrics
    metrics = calculate_metrics(y_true, y_scores, threshold=config.evaluation.get('threshold', 0.5))
    cm_stats = confusion_matrix_stats(y_true, y_scores, threshold=config.evaluation.get('threshold', 0.5))
    metrics.update(cm_stats)
    
    metrics['brier_score'] = brier_score(y_true, y_scores)
    metrics['ece'] = expected_calibration_error(y_true, y_scores)
    
    if args.use_mc_dropout:
        metrics['mean_uncertainty'] = float(np.mean(y_variances))
    
    top_k_values = config.evaluation.get('top_k', [10, 50, 100])
    recall_at_k = calculate_top_k_recall(y_true, y_scores, top_k_values)
    metrics.update(recall_at_k)
    
    print("Metrics:", metrics)
    
    # 2. Plotting (Basic)
    plotter = Plotter(output_dir=config.evaluation.get('output_dir', 'reports/results'))
    try:
        plotter.plot_roc_curve(y_true, y_scores)
        plotter.plot_pr_curve(y_true, y_scores)
        plotter.plot_calibration_curve(y_true, y_scores)
    except Exception as e:
        print(f"Plotting error: {e}")
    
    # 3. Bayesian Ranking
    print("Performing Ranking...")
    # Reconstruct DataFrame
    feature_names = config.numerical_features + config.categorical_features
    if X_all.shape[1] == len(feature_names):
        processed_feature_names = feature_names
    else:
        processed_feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]
        
    features_df = pd.DataFrame(X_all, columns=processed_feature_names)
    
    # Add variant/gene IDs if available? 
    # The test_loader only yields X, y. IDs are lost.
    # For real evaluation, we need IDs. 
    # The Ranker uses dummy IDs if not provided.
    
    # Apply Bayesian Ranking
    bayesian = BayesianRanker()
    # If using MC Dropout, pass variances. Else pass None.
    var_arg = y_variances if args.use_mc_dropout else None
    
    rank_res = bayesian.rank(y_scores, var_arg)
    
    # Update features df with ranking info
    features_df['score'] = y_scores
    features_df['true_label'] = y_true
    features_df['posterior_mean'] = rank_res['posterior_mean']
    features_df['posterior_lower'] = rank_res['posterior_lower']
    features_df['posterior_upper'] = rank_res['posterior_upper']
    features_df['rank'] = rank_res['ranks']
    
    if args.use_mc_dropout:
        features_df['uncertainty_var'] = y_variances

    # Save Ranked List
    output_dir = config.evaluation.get('output_dir', 'reports/results')
    os.makedirs(output_dir, exist_ok=True)
    features_df.to_csv(os.path.join(output_dir, "ranked_mutations.csv"), index=False)
    
    # 4. Biological (Enrichment) - Mocking IDs
    # See original evaluate.py for ID handling if available
    
    # 5. Interpretability (Skip for Ensemble for now or use KernelExplainer on predict)
    # ...
    
    # 6. Gene Aggregation
    if args.aggregate_genes:
        print("Performing Gene Aggregation...")
        # Since we don't have gene_ids in X, y, we can't do essentially.
        # Unless we load the original dataframe validation set.
        # Limit: train/test split in data_loader returns tensors.
        # Solution: Load original file, and assume test split is identifiable?
        # For simplicity, we skip real aggregation here if gene_ids missing.
        # In a real pipeline, we'd pass IDs through the loader or use indices.
        # We will add a placeholder note.
        print("Warning: Gene IDs currently not propagated through DataLoader. Aggregation requires gene_id column.")
        
    # 7. Report
    report = EvaluationReport(output_dir=output_dir)
    report.add_metrics(metrics)
    report.add_config(config.config)
    report.save()
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
