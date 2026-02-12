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
project-root/
│
├── src/                      # Source code
│   ├── config/               # Configuration files
│   ├── models/               # PyTorch model definitions
│   ├── training/             # Training logic
│   ├── evaluation/           # Metrics and reporting
│   ├── ensemble/             # Stacking ensemble
│   ├── uncertainty/          # MC Dropout implementation
│   ├── graph/                # Graph construction and GNNs
│   ├── aggregation/          # Gene scoring
│   ├── ranking/              # Bayesian ranking
│   └── utils/                # Data loaders and helpers
│
├── data/
│   ├── raw/                  # Input data (not tracked by git)
│   └── processed/            # Processed features and artifacts
│
├── reports/                  # Generated results and figures
├── tests/                    # Unit tests
└── README.md
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

Hyperparameters and paths are managed in `src/config/config.yaml`.

## License

[License Name]
