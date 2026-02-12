# AI-Based Genetic Mutation Prioritization

## Abstract

This project implements an AI-driven approach for prioritizing genetic mutations, specifically focusing on distinguishing pathogenic variants from benign ones. It leverages tabular data (Variant Annotation), Graph Neural Networks (GNNs) for gene interaction modeling, and Ensemble Learning to achieve robust predictions reliability.

## Architecture

The system consists of the following modules:

1. **Data Processing**: Standardization and feature engineering of genomic data.
2. **Models**:
    - **Baselines**: Logistic Regression, MLP.
    - **Ensemble**: Stacking of MLP, XGBoost, and LightGBM.
    - **Graph**: GNNs (GCN/GraphSAGE) for variant-gene interaction.
3. **Uncertainty Estimation**: MC Dropout for epistemic uncertainty quantification.
4. **Ranking**: Bayesian ranking based on prediction confidence intervals.
5. **Aggregation**: Gene-level scoring (mean, max, Bayesian aggregation).

## Project Structure

```
AI-Based-approach-for-prioritization-of-genetic-mutations/
│
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   │   ├── loader.py             # Config loader and validator
│   │   └── config.yaml           # Main configuration file
│   ├── models/                   # PyTorch model definitions
│   │   ├── baseline.py           # Logistic regression baseline
│   │   ├── mlp.py                # Multi-layer perceptron
│   │   └── gnn.py                # Graph neural network models
│   ├── training/                 # Training logic
│   │   └── trainer.py            # Model training orchestration
│   ├── evaluation/               # Metrics and reporting
│   │   ├── eval_metrics.py       # Core evaluation metrics
│   │   ├── eval_plotting.py      # Visualization utilities
│   │   ├── eval_report.py        # Report generation
│   │   ├── ranker.py             # Variant ranking logic
│   │   └── biological.py         # Biological validation
│   ├── ensemble/                 # Stacking ensemble
│   │   └── stacking.py           # Multi-model stacking
│   ├── uncertainty/              # Uncertainty quantification
│   │   └── mc_dropout.py         # Monte Carlo dropout
│   ├── graph/                    # Graph-based modeling
│   │   └── construct.py          # Graph construction utilities
│   ├── aggregation/              # Gene-level scoring
│   │   └── gene_score.py         # Variant-to-gene aggregation
│   ├── ranking/                  # Confidence-aware ranking
│   │   └── bayesian.py           # Bayesian ranking with uncertainty
│   ├── interpretation/           # Model explainability
│   │   └── explainer.py          # SHAP/attention interpretation
│   ├── utils/                    # Data utilities
│   │   ├── data_loader.py        # PyTorch data loading
│   │   ├── dataset.py            # Custom dataset classes
│   │   ├── preprocessing.py      # Feature preprocessing
│   │   └── data_generator.py     # Synthetic data generation
│   ├── main.py                   # CLI entry point
│   ├── train.py                  # Model training script
│   ├── train_ensemble.py         # Ensemble training script
│   ├── evaluate.py               # Evaluation script
│   └── prioritize.py             # Inference/prioritization script
│
├── data/
│   ├── raw/                      # Raw input data (VCF, labels)
│   └── processed/                # Preprocessed features and artifacts
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── Model_Training.ipynb      # Interactive model training
│
├── reports/                      # Generated results and figures
│   ├── figures/                  # Plots and visualizations
│   └── results/                  # Model outputs and logs
│       ├── checkpoints/          # Saved model weights
│       └── logs/                 # Training history
│
├── tests/                        # Unit and integration tests
│   ├── test_aggregation.py       # Gene aggregation tests
│   ├── test_bayesian.py          # Bayesian ranking tests
│   ├── test_ensemble.py          # Ensemble model tests
│   ├── test_graph.py             # Graph construction tests
│   ├── test_uncertainty.py       # MC Dropout tests
│   └── test_config.yaml          # Test configuration
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
├── LICENSE                       # Project license
└── README.md                     # This file
```

## Installation

1. Clone the repository:

    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training

Train a standard MLP model:

```bash
python src/main.py train --config src/config/config.yaml --model mlp --epochs 50
```

Train an ensemble model:

```bash
python src/main.py train-ensemble --config src/config/config.yaml
```

### 2. Evaluation

Evaluate the trained model with advanced metrics (ECE, Brier Score) and ranking:

```bash
python src/main.py evaluate --use_ensemble --use_mc_dropout
```

### 3. prioritization (Inference)

Prioritize mutations in a new dataset:

```bash
python src/main.py prioritize data/raw/new_variants.csv --output reports/results/prioritized.csv
```

## Configuration

All hyperparameters, paths, and experiment settings are managed in [src/config/config.yaml](src/config/config.yaml).

The configuration is loaded via the `Config` class in [src/config/loader.py](src/config/loader.py):

```python
from src.config.loader import Config

config = Config("src/config/config.yaml")
print(config.model['type'])  # Access model configuration
print(config.training['epochs'])  # Access training parameters
```

### Key Configuration Sections

- **data**: Feature definitions, paths, preprocessing options
- **model**: Model architecture and hyperparameters
- **training**: Learning rate, batch size, epochs, optimizer settings
- **evaluation**: Metrics, thresholds, calibration parameters
- **ensemble**: Stacking configuration and base learners
- **uncertainty**: MC Dropout settings
- **graph**: GNN architecture and graph construction
- **aggregation**: Gene-level scoring methods
- **ranking**: Bayesian ranking parameters

## License

[License Name]
