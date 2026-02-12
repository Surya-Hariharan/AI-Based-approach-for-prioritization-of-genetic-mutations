# AI-Based Genetic Mutation Prioritization

## Abstract

This project implements an AI-driven approach for prioritizing genetic mutations, specifically focusing on distinguishing pathogenic variants from benign ones. It leverages deep learning models (MLP), ensemble methods (stacking), uncertainty quantification (MC Dropout, Bayesian ranking), and gene-level aggregation to achieve robust and interpretable predictions.

---

## 🎯 Key Features

- **Deep Learning Models**: MLP with configurable architecture, dropout, and batch normalization
- **Baseline Models**: Logistic regression for performance comparison
- **Ensemble Learning**: Stacking with multiple base learners and meta-learner
- **Uncertainty Quantification**: MC Dropout for epistemic uncertainty estimation
- **Bayesian Ranking**: Confidence-aware variant prioritization with credible intervals
- **Gene-Level Aggregation**: Variant-to-gene score aggregation for biological interpretation
- **Reproducibility**: Centralized seed management and deterministic mode
- **Config-Driven**: Single YAML configuration file controls all experiments

---

## 📁 Project Structure

```
AI-Based-approach-for-prioritization-of-genetic-mutations/
│
├── configs/                      # Configuration files
│   └── config.yaml              # Single source of truth for all parameters
│
├── data/                        # Data storage (strict lifecycle: RAW → INTERIM → PROCESSED)
│   ├── raw/                     # ❌ READ-ONLY: Original datasets (VCF, CSV)
│   │   ├── clinvar_input.vcf
│   │   ├── mutation_impact_dataset.csv
│   │   └── labels.csv
│   ├── interim/                 # ⚙️ DERIVED: Engineered features (not preprocessed)
│   │   └── feature_matrix_raw.csv
│   └── processed/               # ✅ TRAINING-READY: Preprocessed features
│       ├── feature_matrix_processed.csv
│       └── preprocessor.joblib
│
├── notebooks/                   # Experimentation workflows ⭐ START HERE!
│   ├── 00_data_pipeline.ipynb           # ⚠️ RUN FIRST: RAW → INTERIM → PROCESSED
│   ├── 01_data_exploration.ipynb         # EDA and feature analysis
│   ├── 02_baseline_training.ipynb        # Logistic regression baseline
│   ├── 03_mlp_training.ipynb             # Deep learning training
│   ├── 04_ensemble_training.ipynb        # Stacking ensemble
│   ├── 05_uncertainty_analysis.ipynb     # MC Dropout + Bayesian ranking
│   └── 06_gene_level_ranking.ipynb       # Gene aggregation
│
├── src/                         # Source code (import from here!)
│   ├── models/                  # Model architectures (NO training logic)
│   │   ├── baseline.py          # LogisticRegression
│   │   ├── mlp.py              # Multi-layer perceptron
│   │   └── gnn.py              # Graph neural network
│   │
│   ├── preprocessing/           # Data processing
│   │   ├── data_loader.py      # DataLoader creation
│   │   ├── preprocessing.py    # Feature engineering
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── validation.py       # Data validation utilities
│   │   └── pipeline.py         # RAW → INTERIM → PROCESSED pipeline
│   │
│   ├── evaluation/              # Metrics and visualization
│   │   ├── metrics.py          # ROC-AUC, PR-AUC, F1
│   │   ├── plotting.py         # Plotter class
│   │   ├── ranking_metrics.py  # Ranking evaluation
│   │   ├── reporting.py        # Report generation
│   │   └── calibration.py      # Calibration analysis
│   │
│   ├── uncertainty/             # Uncertainty quantification
│   │   ├── mc_dropout.py       # MC Dropout estimator
│   │   └── bayesian_ranking.py # Bayesian ranker
│   │
│   ├── ensemble/                # Ensemble methods
│   │   └── stacking.py         # Stacking ensemble
│   │
│   ├── aggregation/             # Gene-level aggregation
│   │   └── gene_score.py       # GeneAggregator
│   │
│   └── utils/                   # Utilities
│       ├── seed.py             # Reproducibility (set_seed)
│       └── config.py           # Config management
│
├── reports/                     # Output storage
│   ├── figures/                 # Plots and visualizations
│   └── results/                 # Metrics, checkpoints, rankings
│       ├── checkpoints/         # Model weights (.pth, .joblib)
│       └── logs/               # Training history (.json)
│
├── tests/                       # Testing utilities
│
├── REFACTORING_SUMMARY.md      # Detailed architectural documentation
├── QUICK_REFERENCE.md          # Quick start guide
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo_url>
cd AI-Based-approach-for-prioritization-of-genetic-mutations

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks

Open Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

**Recommended execution order:**
0. [00_data_pipeline.ipynb](notebooks/00_data_pipeline.ipynb) - **⚠️ RUN FIRST!** Executes RAW → INTERIM → PROCESSED pipeline
1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) - Understand your data
2. [02_baseline_training.ipynb](notebooks/02_baseline_training.ipynb) - Establish baseline
3. [03_mlp_training.ipynb](notebooks/03_mlp_training.ipynb) - Train deep learning model
4. [04_ensemble_training.ipynb](notebooks/04_ensemble_training.ipynb) - Combine models
5. [05_uncertainty_analysis.ipynb](notebooks/05_uncertainty_analysis.ipynb) - Quantify uncertainty
6. [06_gene_level_ranking.ipynb](notebooks/06_gene_level_ranking.ipynb) - Generate gene rankings

### 3. Data Lifecycle

This project enforces strict data separation: **RAW → INTERIM → PROCESSED**

```
data/
├── raw/          # ❌ READ-ONLY: Original untouched data
├── interim/      # ⚙️ DERIVED: Engineered features (not scaled)
└── processed/    # ✅ TRAINING-READY: Final processed data
```

**Key Rules**:
- ❌ **NEVER WRITE** to `data/raw/` after initial placement
- ✅ Run `00_data_pipeline.ipynb` to process data
- ✅ All training loads from `data/processed/`
- ✅ Validation enforced automatically

See [DATA_LIFECYCLE.md](DATA_LIFECYCLE.md) for complete documentation.

### 4. Configuration

Edit [configs/config.yaml](configs/config.yaml) to customize:
- Data paths
- Model hyperparameters
- Training settings
- Uncertainty estimation parameters

---

## 📊 Models and Methods

### 1. Baseline Models
- **Logistic Regression**: Single-layer linear model for binary classification
- Serves as performance baseline for deep learning models

### 2. Deep Learning (MLP)
- **Architecture**: Configurable hidden layers with dropout and batch normalization
- **Training**: Early stopping with validation monitoring
- **Optimization**: Adam optimizer with BCE loss

### 3. Ensemble Learning
- **Method**: Stacking with multiple base learners
- **Base Models**: MLP, XGBoost, LightGBM
- **Meta-Learner**: Logistic Regression
- **Benefit**: Improved robustness and performance

### 4. Uncertainty Quantification
- **MC Dropout**: Bayesian approximation via stochastic forward passes
- **Output**: Predictive mean and variance for each variant
- **Use Case**: Identifying low-confidence predictions

### 5. Bayesian Ranking
- **Method**: Posterior inference with Beta-Binomial conjugate prior
- **Output**: Ranked variants with credible intervals
- **Benefit**: Incorporates uncertainty into prioritization

### 6. Gene-Level Aggregation
- **Methods**: Max, mean, median aggregation
- **Input**: Variant-level scores
- **Output**: Gene-level priority scores
- **Use Case**: Identifying high-risk genes

---

## 📈 Evaluation Metrics

- **ROC-AUC**: Overall discrimination performance
- **PR-AUC**: Performance on imbalanced data
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Classification breakdown
- **Calibration**: Reliability of predicted probabilities

---

## 🔬 Research Features

### Reproducibility
```python
from src.utils.seed import set_seed

set_seed(42)  # Ensures reproducible results
```

### Config-Driven Experiments
```python
from src.utils.config import Config

config = Config('configs/config.yaml')
batch_size = config.training['batch_size']
```

### Clean Module Imports
```python
from src.models import MLP, LogisticRegression
from src.preprocessing import get_data_loaders
from src.evaluation import calculate_metrics, Plotter
from src.uncertainty import MCDropoutEstimator, BayesianRanker
```

---

## 📚 Documentation

- **[DATA_LIFECYCLE.md](DATA_LIFECYCLE.md)**: Data pipeline and lifecycle management
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**: Comprehensive architectural documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Quick start guide and API reference

---

## 🧪 Testing

Generate dummy data for testing:
```bash
python tests/create_dummy_data.py
```

---

## 🏗️ Architecture Principles

✅ **Separation of Concerns**: Architecture in `src/`, experiments in `notebooks/`  
✅ **Config-Driven**: Single source of truth in `configs/config.yaml`  
✅ **Reproducibility**: Centralized seed management  
✅ **Modularity**: Clean imports via `__init__.py`  
✅ **Research-Grade**: Publication-ready code quality  

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

This project follows a clean research architecture. When contributing:
1. Add model architectures to `src/models/` (NO training logic)
2. Create experiments in `notebooks/`
3. Update `configs/config.yaml` for new parameters
4. Ensure reproducibility with `set_seed(42)`

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Happy Mutation Prioritizing! 🧬**

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
