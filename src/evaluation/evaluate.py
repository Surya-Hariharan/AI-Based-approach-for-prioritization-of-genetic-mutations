import argparse
import torch
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config.data_config import Config
from src.utils.data_loader import get_data_loaders
from src.models.baseline import LogisticRegression
from src.models.mlp import MLP
from src.evaluation.metrics import calculate_metrics, calculate_top_k_recall, confusion_matrix_stats
from src.evaluation.plotting import Plotter
from src.evaluation.report import EvaluationReport
from src.evaluation.ranker import Ranker
from src.evaluation.biological import BiologicalEvaluator
from src.interpretation.explainer import ModelInterpreter

def main():
    parser = argparse.ArgumentParser(description="Evaluate Genetic Mutation Prioritization Model")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/best_model.pth", help="Path to trained model")
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data (Test Set)
    # We re-use get_data_loaders but strictly we'd want to just load test, 
    # but for simplicity we call the main loader which ensures consistent splitting.
    _, _, test_loader, input_dim = get_data_loaders(config)
    
    # Load Model
    model_type = config.model['type']
    if model_type == "baseline":
        model = LogisticRegression(input_dim)
    elif model_type == "mlp":
        model = MLP(input_dim, config.model['mlp']['hidden_layers'], config.model['mlp']['dropout'])
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run Inference
    y_true = []
    y_scores = []
    X_all = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs)
            
            y_true.extend(y.cpu().numpy().flatten())
            y_scores.extend(probs.cpu().numpy().flatten())
            X_all.extend(X.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    X_all = np.array(X_all)
    
    # 1. Base Metrics
    metrics = calculate_metrics(y_true, y_scores, threshold=config.evaluation.get('threshold', 0.5))
    cm_stats = confusion_matrix_stats(y_true, y_scores, threshold=config.evaluation.get('threshold', 0.5))
    metrics.update(cm_stats)
    
    top_k_values = config.evaluation.get('top_k', [10, 50, 100])
    recall_at_k = calculate_top_k_recall(y_true, y_scores, top_k_values)
    metrics.update(recall_at_k)
    
    print("Metrics Calculated:", metrics)
    
    # 2. Plotting
    plotter = Plotter(output_dir=config.evaluation.get('output_dir', 'reports/results'))
    plotter.plot_roc_curve(y_true, y_scores)
    plotter.plot_pr_curve(y_true, y_scores)
    plotter.plot_calibration_curve(y_true, y_scores)
    
    # 3. Ranking
    # For demo, we reconstruct a DataFrame. In prod, we'd pass metadata through loader.
    # Here we just use X_all values as features.
    feature_names = config.numerical_features + config.categorical_features
    # Note: One-hot encoding increases feature count, so names might assume original?
    # Actually ModelInput is processed. We don't have exact column names for processed data here easily
    # without the preprocessor. Ideally we save feature names in preprocessor.
    # For now, we'll label generic features for interpretabiltiy if size mismatches.
    
    if X_all.shape[1] == len(feature_names):
        processed_feature_names = feature_names
    else:
        # Fallback for OHE expansion
        processed_feature_names = [f"feat_{i}" for i in range(X_all.shape[1])]

    features_df = pd.DataFrame(X_all, columns=processed_feature_names)
    ranker = Ranker(output_dir=config.evaluation.get('output_dir', 'reports/results'))
    ranked_df = ranker.rank_mutations(y_scores, features_df, target_labels=y_true)
    ranker.save_ranked_list(ranked_df)
    
    # 4. Biological Evaluation (Enrichment)
    # Mocking pathogenic variants as valid positives in test set for demonstration
    # In reality, this would be an external set ID list.
    pathogenic_ids = set(ranked_df[ranked_df['true_label'] == 1]['variant_id'].values)
    bio_eval = BiologicalEvaluator(pathogenic_variants=pathogenic_ids)
    enrichment = bio_eval.enrichment_analysis(ranked_df, top_k=50)
    metrics['enrichment_factor_top50'] = enrichment
    
    # 5. Interpretation (SHAP)
    # Use background samples from config
    n_bg = config.interpretation.get('background_samples', 100)
    bg_data = torch.tensor(X_all[:n_bg], dtype=torch.float32).to(device)
    
    explainer = ModelInterpreter(model, bg_data, processed_feature_names, output_dir=config.interpretation.get('output_dir', 'reports/figures'))
    # Explain a subset of test data
    explainer.explain_summary(X_all[:n_bg])
    
    # 6. Report Generation
    report = EvaluationReport(output_dir=config.evaluation.get('output_dir', 'reports/results'))
    report.add_metrics(metrics)
    report.add_config(config.config)
    report.save()

if __name__ == "__main__":
    main()
