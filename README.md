# AI-Based Genetic Mutation Prioritization  
**Machine Learning Platform for Pathogenic Variant Classification**  
*Research-Grade Pipeline with Production-Ready Web Interface*

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)
![Machine Learning](https://img.shields.io/badge/ML-Production%20Ready-success)

---

## 🧬 Problem Statement

Develop an AI-based system to **prioritize genetic mutations** and **distinguish pathogenic variants from benign ones** in clinical genomics applications:

- 🔬 **Variant Classification**: Binary prediction of pathogenic vs benign mutations
- 📊 **Uncertainty Quantification**: Confidence scores for clinical decision making
- 🧬 **Gene-Level Ranking**: Aggregated scores for identifying high-risk genes
- ⚙️ **Clinical Integration**: Production-ready API for healthcare systems
- 📈 **Research Platform**: Modular pipeline for genomics research

**Clinical Impact**: Enable precision medicine by accurately identifying disease-causing genetic variants from large-scale genomic data.

---

## 💡 Solution Overview

**AI-Based Genetic Mutation Prioritization** is a comprehensive machine learning platform that combines multiple AI approaches for robust variant classification:

- **Multi-Model Architecture**: MLP, ensemble stacking, and baseline models
- **Uncertainty Estimation**: MC Dropout and Bayesian ranking for confidence assessment
- **Research Pipeline**: End-to-end notebooks from data processing to gene ranking
- **Production Interface**: Flask web application with RESTful API
- **Clinical-Ready**: Reproducible, traceable, and interpretable predictions

**Target Applications:**  
- 🏥 Clinical diagnostics and genetic counseling
- 🔬 Research genomics and variant discovery
- 💊 Precision medicine and drug development
- 📋 Population health and genetic screening

---

## ⚙️ Key Features

### Core ML Capabilities
- 🧠 **Deep Learning Models** - MLP with configurable architecture and dropout
- 📈 **Ensemble Methods** - Stacking with multiple base learners
- 🎯 **Baseline Models** - Logistic regression for performance comparison
- 🔮 **Uncertainty Quantification** - MC Dropout epistemic uncertainty estimation
- 📊 **Bayesian Ranking** - Confidence-aware prioritization with credible intervals
- 🧬 **Gene-Level Aggregation** - Variant-to-gene score aggregation

### Production Features
- 🌐 **Web Application** - Modern Flask interface with real-time predictions
- 📤 **Batch Processing** - CSV file upload for multiple variant analysis
- 🔌 **RESTful API** - Programmatic access for system integration
- 📱 **Responsive Design** - Mobile-friendly interface with drag & drop
- ⚡ **Real-time Results** - Instant predictions with confidence visualization
- 🎯 **Model Selection** - Choose between MLP, baseline, or ensemble models

### Research Features
- 🔄 **Reproducible Pipeline** - Centralized seed management and deterministic mode
- ⚙️ **Config-Driven** - Single YAML configuration file controls all experiments
- 📓 **Jupyter Notebooks** - Complete workflow from data to deployment
- 📋 **Comprehensive Evaluation** - ROC-AUC, PR-AUC, calibration analysis
- 🎨 **Rich Visualizations** - Publication-ready plots and metrics
- 📊 **Performance Tracking** - Detailed logging and checkpoint management

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web Interface                         │
│         (Flask App - Interactive Predictions)           │
│  • File Upload (CSV/VCF) • Model Selection              │
│  • Real-time Results    • Confidence Visualization      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST API
                       ↓
┌─────────────────────────────────────────────────────────┐
│                   Flask Backend                         │
│  • Model Loading & Caching                              │
│  • Input Validation & Preprocessing                     │
│  • Error Handling & logging                             │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ↓                             ↓
┌──────────────┐              ┌──────────────┐
│   Data       │              │   Model      │
│ Processing   │              │  Inference   │
└──────┬───────┘              └──────┬───────┘
       │                             │
       ↓                             ↓
┌──────────────┐              ┌──────────────────┐
│ Feature      │              │  Model Zoo       │
│ Engineering  │              │  • MLP (PyTorch) │
│ Pipeline     │              │  • Baseline      │
└──────┬───────┘              │  • Ensemble      │
       │                      └──────┬───────────┘
       ↓                             │
┌──────────────────────────────────────────────┐
│           Research Pipeline                  │
│  • Data Lifecycle (RAW → INTERIM → PROC)     │
│  • Jupyter Notebooks (00-06)                 │
│  • Config Management (YAML)                  │
│  • Reproducible Training                     │
└──────────────────┬───────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────┐
│          Model Training & Evaluation         │
│  • Cross-validation • Hyperparameter Tuning  │
│  • Uncertainty Estimation • Calibration      │
│  • Performance Metrics • Visualization       │
└──────────────────┬───────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────┐
│            Model Artifacts                   │
│  • Trained Models (.pth, .joblib)            │
│  • Preprocessors • Performance Reports       │
│  • Gene Rankings • Uncertainty Estimates     │
└──────────────────────────────────────────────┘
```

---

## 🖥 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Flask 2.0+ | Production API with auto-documentation |
| **Deep Learning** | PyTorch 2.0+ | Neural network training and inference |
| **Machine Learning** | scikit-learn | Ensemble methods and baseline models |
| **Data Processing** | pandas, NumPy | Feature engineering and data manipulation |
| **Visualization** | Matplotlib, Seaborn | Performance analysis and reporting |
| **Configuration** | YAML | Centralized experiment management |
| **Notebooks** | Jupyter | Interactive research and development |
| **Serialization** | joblib | Model persistence and loading |
| **Frontend** | HTML5, CSS3, JavaScript | Modern responsive web interface |
| **Deployment** | Docker (ready) | Containerization and scaling |

---

## 📂 Project Structure

```
AI-Based-approach-for-prioritization-of-genetic-mutations/
│
├── 🌐 Web Application (Production Interface)
│   ├── backend/                # Flask REST API Server
│   │   ├── app.py             # Main Flask application with endpoints
│   │   ├── config.py          # Configuration management
│   │   └── __init__.py        # Package initialization
│   │
│   ├── frontend/              # Modern Web Interface
│   │   ├── templates/
│   │   │   └── index.html     # Responsive UI with drag & drop
│   │   └── static/
│   │       ├── css/style.css  # Professional styling
│   │       └── js/app.js      # Interactive JavaScript
│   │
│   └── main.py               # Application entry point (multi-mode)
│
├── 📓 Research Pipeline (Jupyter Notebooks)
│   └── notebooks/
│       ├── 00_data_pipeline.ipynb      # ⚠️ RUN FIRST: Data processing
│       ├── 01_data_exploration.ipynb   # EDA and visualization
│       ├── 02_baseline_training.ipynb  # Logistic regression
│       ├── 03_mlp_training.ipynb       # Deep learning (MLP)
│       ├── 04_ensemble_training.ipynb  # Stacking ensemble
│       ├── 05_uncertainty_analysis.ipynb # MC Dropout + Bayesian
│       └── 06_gene_level_ranking.ipynb # Gene aggregation
│
├── 🔧 Configuration & Data Management
│   ├── configs/
│   │   └── config.yaml        # Centralized configuration
│   │
│   ├── data/
│   │   ├── raw/              # ❌ READ-ONLY: Original datasets
│   │   │   ├── clinvar_input.vcf
│   │   │   ├── labels.csv
│   │   │   └── mutation_impact_dataset.csv
│   │   ├── interim/          # ⚙️ DERIVED: Feature engineering
│   │   │   └── feature_matrix_raw.csv
│   │   ├── processed/        # ✅ TRAINING-READY: Final data
│   │   │   ├── feature_matrix_processed.csv
│   │   │   └── preprocessor.joblib
│   │   └── uploads/          # 📤 Web app file uploads
│       
├── 🧠 Source Code (Core ML Pipeline)
│   └── src/
│       ├── models/              # Neural architectures
│       │   ├── baseline.py      # Logistic regression baseline
│       │   ├── mlp.py          # Multi-layer perceptron
│       │   └── gnn.py          # Graph neural networks
│       │
│       ├── preprocessing/       # Data processing pipeline
│       │   ├── data_loader.py   # PyTorch DataLoader creation
│       │   ├── preprocessing.py # Feature engineering
│       │   ├── dataset.py       # Custom PyTorch Dataset
│       │   ├── pipeline.py      # Processing pipeline
│       │   └── validation.py    # Data validation
│       │
│       ├── evaluation/          # Metrics and visualization
│       │   ├── metrics.py       # ROC-AUC, PR-AUC, F1
│       │   ├── plotting.py      # Publication-ready plots
│       │   ├── calibration.py   # Model calibration analysis
│       │   ├── ranking_metrics.py # Gene ranking evaluation
│       │   └── reporting.py     # Results reporting
│       │
│       ├── uncertainty/         # Confidence estimation
│       │   ├── mc_dropout.py    # Monte Carlo Dropout
│       │   └── bayesian_ranking.py # Bayesian ranking
│       │
│       ├── ensemble/            # Multi-model approaches
│       │   └── stacking.py      # Ensemble stacking methods
│       │
│       ├── aggregation/         # Gene-level analysis
│       │   └── gene_score.py    # Variant-to-gene score aggregation
│       │
│       └── utils/               # Core utilities
│           ├── seed.py          # Reproducibility management
│           ├── config.py        # Configuration utilities
│           └── data_generator.py # Synthetic data generation
│
├── 📊 Outputs & Results
│   └── reports/
│       └── results/            # Model outputs and rankings
│           ├── ranked_genes.csv # Gene prioritization results
│           └── checkpoints/    # Trained model files
│               ├── baseline_model.pth
│               ├── mlp_best.pth
│               └── ensemble_model.joblib
│
├── 🧪 Project Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── setup.py               # Package installation
│   ├── .env.example          # Environment configuration
│   ├── .gitignore           # Git ignore patterns
│   ├── LICENSE              # MIT License
│   └── README.md            # This comprehensive guide
---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations.git
cd AI-Based-approach-for-prioritization-of-genetic-mutations

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Quick Start (Web Application)

**Launch the web interface immediately:**

```bash
# Method 1: Direct launch
python main.py --mode web

# Method 2: Development mode with auto-reload
python main.py --mode web --debug

# Method 3: Production mode
python main.py --mode web --config production --host 0.0.0.0
```

🌐 **Access**: `http://localhost:5000`

**Features Available:**
- 📤 **Drag & Drop Upload**: Batch analyze CSV files
- ⌨️ **Manual Input**: Test individual mutations
- 🎯 **Model Selection**: Choose MLP, Baseline, or Ensemble
- 📊 **Real-time Visualization**: Confidence scores and probability bars
- 📈 **Performance Dashboard**: System statistics and model metrics

### 3️⃣ Research Workflow

**Launch research environment:**

```bash
# Start Jupyter for research pipeline
python main.py --mode research

# Or manually
jupyter notebook
```

**Notebook Execution Order:**
```bash
# Essential workflow:
00_data_pipeline.ipynb       # ⚠️ RUN FIRST: Data processing
01_data_exploration.ipynb    # Understand your data
02_baseline_training.ipynb   # Establish baseline performance  
03_mlp_training.ipynb        # Train deep learning model
04_ensemble_training.ipynb   # Combine models for better performance
05_uncertainty_analysis.ipynb # Quantify prediction confidence
06_gene_level_ranking.ipynb  # Generate gene-level priorities
```

### 4️⃣ Configuration

**Environment Setup:**
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

**Key Settings:**
- `FLASK_ENV`: development/production
- `USE_GPU`: Enable/disable GPU acceleration
- `DEFAULT_MODEL`: mlp/baseline/ensemble

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

### 5. Configuration

Edit [configs/config.yaml](configs/config.yaml) to customize:
- Data paths
- Model hyperparameters
- Training settings
- Uncertainty estimation parameters

---

## � Web API Documentation

### Base URL
```
http://localhost:5000
```

### 🔐 Authentication
Secure endpoints (future): Bearer token authentication
```
Authorization: Bearer <your_token>
```

---

### Endpoints

#### 1. Health Check
Monitor system health and loaded models.

**Request:**
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["mlp", "baseline", "ensemble"],
  "device": "cuda"
}
```

#### 2. Single Mutation Prediction
Predict pathogenicity for a single mutation.

**Request:**
```http
POST /api/predict
Content-Type: application/json

{
  "features": [0.8, 0.3, 0.9, 0.2, 0.7],
  "model": "mlp"
}
```

**Response:**
```json
{
  "success": true,
  "model": "mlp",
  "prediction": {
    "prediction": "Pathogenic",
    "probability": 0.89,
    "confidence": "High",
    "pathogenic_score": 0.89,
    "benign_score": 0.11
  }
}
```

#### 3. Batch File Upload
Analyze multiple mutations from CSV file.

**Request:**
```http
POST /api/predict
Content-Type: multipart/form-data

file: mutations.csv
model: ensemble
```

**Response:**
```json
{
  "success": true,
  "model": "ensemble",
  "predictions": [
    {
      "prediction": "Pathogenic",
      "probability": 0.92,
      "confidence": "High"
    },
    {
      "prediction": "Benign",
      "probability": 0.15,
      "confidence": "High"
    }
  ],
  "count": 2
}
```

#### 4. System Statistics
Get model performance and system metrics.

**Request:**
```http
GET /api/stats
```

**Response:**
```json
{
  "models_available": 3,
  "models": ["mlp", "baseline", "ensemble"],
  "total_genes_ranked": 1234,
  "top_genes": [
    {"gene": "BRCA1", "mean_score": 0.95},
    {"gene": "TP53", "mean_score": 0.91}
  ]
}
```

---

## 🧠 Machine Learning Models

---

### 1. 🎯 Baseline Models
- **Logistic Regression**: Linear model for binary pathogenicity classification
- **Purpose**: Performance baseline and interpretability reference
- **Training**: Regularized with cross-validation
- **Output**: Probability scores with decision boundary

### 2. 🧠 Deep Learning (MLP)
- **Architecture**: Multi-layer perceptron with configurable hidden layers
- **Features**: Dropout, batch normalization, early stopping
- **Optimization**: Adam optimizer with binary cross-entropy loss
- **Training**: GPU-accelerated with validation monitoring
- **Benefits**: Captures complex non-linear relationships

### 3. 🚀 Ensemble Learning
- **Method**: Stacking ensemble with multiple base learners
- **Base Models**: MLP, XGBoost, LightGBM, Random Forest
- **Meta-Learner**: Logistic regression combiner
- **Benefits**: Improved robustness and reduced overfitting
- **Performance**: Typically best overall accuracy

### 4. 🔮 Uncertainty Quantification
- **MC Dropout**: Bayesian approximation via stochastic forward passes
- **Output**: Predictive mean and confidence intervals
- **Clinical Value**: Identifies low-confidence predictions requiring expert review
- **Implementation**: Multiple forward passes with dropout enabled

### 5. 📊 Bayesian Ranking
- **Method**: Beta-Binomial conjugate prior for posterior inference
- **Output**: Ranked variants with credible intervals
- **Benefits**: Incorporates prediction uncertainty into prioritization
- **Use Case**: Clinical decision support with confidence bounds

### 6. 🧬 Gene-Level Aggregation
- **Input**: Variant-level pathogenicity scores
- **Methods**: Maximum, mean, median aggregation strategies
- **Output**: Gene-level priority rankings
- **Application**: Identifying high-risk genes for targeted screening

---

## 📈 Performance Metrics

### Classification Metrics
- 🎯 **ROC-AUC**: Overall discrimination between pathogenic/benign
- 📊 **PR-AUC**: Performance on imbalanced clinical datasets
- ⚖️ **F1 Score**: Harmonic mean of precision and recall
- 📋 **Confusion Matrix**: Detailed classification breakdown
- 🎪 **Calibration**: Reliability of predicted probabilities for clinical use

### Confidence Metrics
- 🔮 **Epistemic Uncertainty**: Model confidence via MC Dropout
- 📏 **Confidence Intervals**: Bayesian credible intervals
- ⚡ **Prediction Speed**: Sub-second inference for real-time use
- 🎯 **Calibration Error**: Difference between predicted and actual probabilities

---

## 🎯 Clinical Use Cases

### 🏥 Clinical Diagnostics
- **Variant Interpretation**: Prioritize variants in patient genetic tests
- **Diagnostic Support**: Assist geneticists in variant classification
- **Rare Disease**: Identify pathogenic variants in undiagnosed patients
- **Inherited Cancer**: Screen for cancer predisposition variants

### 🔬 Research Applications
- **Population Studies**: Large-scale variant effect analysis
- **Drug Development**: Identify targets for precision medicine
- **Functional Studies**: Prioritize variants for experimental validation
- **Biobank Mining**: Discover disease associations in large cohorts

### 🧬 Precision Medicine
- **Treatment Selection**: Guide therapy based on genetic profile
- **Risk Stratification**: Assess disease susceptibility
- **Pharmacogenomics**: Predict drug response and adverse effects
- **Prevention**: Identify high-risk individuals for screening

---

## ⚙️ Configuration Guide

### Environment Setup

**Create `.env` file** (for web app):
```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Model Settings
DEFAULT_MODEL=mlp
BATCH_SIZE=32
MAX_FILE_SIZE=16777216  # 16MB

# GPU Settings (optional)
CUDA_VISIBLE_DEVICES=0
USE_GPU=true
```

**Edit `configs/config.yaml`** for research:
```yaml
model:
  type: 'mlp'
  mlp:
    hidden_layers: [256, 128, 64]
    dropout: 0.3
    activation: 'relu'

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping: 10

data:
  chunk_size: 1000
  test_split: 0.2
  val_split: 0.1
```

### Research Reproducibility
```python
# Ensure reproducible results
from src.utils.seed import set_seed
set_seed(42)

# Load configuration
from src.utils.config import Config
config = Config('configs/config.yaml')

# Clean imports
from src.models import MLP, LogisticRegression
from src.preprocessing import get_data_loaders
from src.evaluation import calculate_metrics, Plotter
```

---

## � Docker Deployment

### Standard Deployment

**Build Image:**
```bash
docker build -t genetic-mutation-ai:latest .
```

**Run Container:**
```bash
docker run -d \
  --name genetic-ai \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  genetic-mutation-ai:latest
```

### GPU-Enabled Deployment

**Requirements:** NVIDIA Docker Runtime

**Run with GPU:**
```bash
docker run -d \
  --name genetic-ai-gpu \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  genetic-mutation-ai:latest
```

### Docker Compose (Full Stack)

**Create `docker-compose.yml`:**
```yaml
version: '3.8'

services:
  genetic-ai:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
    environment:
      - FLASK_ENV=production
      - USE_GPU=true
    depends_on:
      - postgres
  
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: genetic_mutations
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

**Deploy:**
```bash
docker-compose up -d
```

---

## 🚨 Troubleshooting

### Common Issues

#### ❌ No Models Found Error
**Problem:** "No models found. Please train models first."

**Solution:**
```bash
# Train at least one model using notebooks:
jupyter notebook
# Run: 02_baseline_training.ipynb OR 03_mlp_training.ipynb

# Verify model files exist:
ls reports/results/checkpoints/
# Should see: baseline_model.pth, mlp_best.pth, ensemble_model.joblib
```

#### ❌ Import/Module Errors
**Problem:** `ModuleNotFoundError` or import issues

**Solution:**
```bash
# Ensure correct environment
which python
pip list | grep torch

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Add project to Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### ❌ CUDA/GPU Issues
**Problem:** GPU not detected or CUDA errors

**Solution:**
```python
# Check PyTorch CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Force CPU if needed (in config.yaml):
device: 'cpu'
```

#### ❌ Web App Port Conflicts
**Problem:** "Address already in use"

**Solution:**
```bash
# Kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

#### ❌ Large File Upload Issues
**Problem:** File upload fails for large CSV files

**Solution:**
```python
# Increase limit in app.py:
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB
```

---

## 📚 Documentation

---

### 📖 Complete Guides
- **[WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)**: Complete API documentation and examples
- **[QUICK_START.md](QUICK_START.md)**: 30-second web app setup guide
- **[DATA_LIFECYCLE.md](DATA_LIFECYCLE.md)**: Data pipeline and lifecycle management
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**: Architectural documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Research pipeline reference

### 📊 Interactive Documentation
- **Web Interface**: `http://localhost:5000` (when running)
- **Jupyter Notebooks**: Complete research workflow with examples
- **Configuration**: `configs/config.yaml` with detailed comments

---

## 🧪 Testing & Validation

### Sample Data

**Test the web app** with provided sample:
```bash
# Use the included test file
open data/uploads/sample_test.csv
# Contains 5 mutations with varying pathogenicity scores
```

**Generate synthetic data** for research:
```python
# In Python/Jupyter:
from src.utils.data_generator import generate_synthetic_mutation_data

generate_synthetic_mutation_data(
    n_samples=1000,
    output_path='data/test_mutations.csv',
    include_labels=True,
    random_seed=42
)
```

### Model Validation

**Cross-validation** is built into the research pipeline:
```python
# In notebooks, models are validated with:
# - Train/validation/test splits
# - Stratified cross-validation
# - Performance tracking across folds
# - Bootstrap confidence intervals
```

**Performance benchmarks:**
- 🎯 **ROC-AUC**: >0.85 (clinical threshold)
- 📊 **PR-AUC**: >0.80 (imbalanced data performance)
- ⚖️ **F1 Score**: >0.75 (balanced precision/recall)
- 🎪 **Calibration**: <0.1 Brier score

---

## 🔒 Security & Best Practices

### Production Security Checklist
- ✅ **Input Validation**: All API inputs validated and sanitized
- ✅ **File Security**: Upload restrictions (CSV/VCF only, size limits)
- ✅ **Error Handling**: Graceful degradation with informative messages
- ✅ **Resource Limits**: Memory and compute resource management
- ✅ **Temporary Cleanup**: Automatic cleanup of uploaded files
- 🔲 **Authentication**: Bearer token support (configurable)
- 🔲 **Rate Limiting**: API request throttling (recommend Nginx/proxy)
- 🔲 **HTTPS/TLS**: SSL encryption for production deployment

### Data Privacy
- 📁 **Local Processing**: All computation happens locally
- 🔒 **No Data Storage**: Uploaded files automatically deleted
- 🧬 **HIPAA Considerations**: Suitable for clinical data with proper deployment
- 🔐 **Model Security**: Trained models stored locally

### Recommended Production Setup
```bash
# Use gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Behind reverse proxy (nginx)
# Enable HTTPS, rate limiting, load balancing
```

---

## 🚀 Future Enhancements

### 🧬 Model Improvements
- [ ] **Graph Neural Networks**: Incorporate protein structure information
- [ ] **Multi-modal Learning**: Combine sequence, structure, and functional data
- [ ] **Active Learning**: Iteratively improve with expert annotations
- [ ] **Federated Learning**: Collaborative training across institutions
- [ ] **Transformer Models**: Attention-based sequence modeling

### 🌐 Interface Enhancements
- [ ] **Mobile App**: Native iOS/Android interface
- [ ] **Browser Extension**: In-page variant annotation
- [ ] **API Gateway**: Enterprise-grade API management
- [ ] **Real-time Dashboard**: Live monitoring and analytics
- [ ] **Integration APIs**: FHIR/HL7 healthcare standards

### 🏥 Clinical Features
- [ ] **EHR Integration**: Electronic health record connectivity
- [ ] **Report Generation**: Automated clinical reports
- [ ] **Decision Support**: Treatment recommendation system
- [ ] **Population Analytics**: Large-scale population health insights
- [ ] **Regulatory Compliance**: FDA/CE marking pathway

### 🔬 Research Tools
- [ ] **AutoML Pipeline**: Automated model selection and tuning
- [ ] **Experiment Tracking**: MLflow integration
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Interpretability**: SHAP/LIME explanation tools
- [ ] **Benchmarking**: Standardized evaluation datasets

---

## 🏗️ Architecture Principles

### Research-Grade Design
✅ **Modular Architecture**: Clean separation of concerns  
✅ **Config-Driven**: Single YAML configuration source  
✅ **Reproducible**: Centralized seed management and deterministic mode  
✅ **Version Control**: Git-friendly structure with clear data lifecycle  
✅ **Documentation**: Comprehensive guides and API documentation  

### Production-Ready Features
✅ **Web Interface**: Modern Flask application with REST API  
✅ **Scalable**: Async-ready architecture for high concurrency  
✅ **Monitoring**: Health checks and performance metrics  
✅ **Error Handling**: Graceful degradation and informative messages  
✅ **Docker Ready**: Containerization for easy deployment  

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

### Commercial Use
- ✅ **Academic Research**: Freely available for academic and research use
- ✅ **Clinical Applications**: Suitable for healthcare with proper validation
- ✅ **Commercial Development**: Permissive license allows commercial use
- ⚠️ **Regulatory Note**: Clinical deployment requires appropriate regulatory approval

---

## 👥 Team & Contributors

Built with ❤️ for precision medicine and genomics research:

- **Surya Hariharan** - Project Lead & Developer  
  - [GitHub](https://github.com/Surya-Hariharan)
  - Architecture design, ML pipeline, web application

### Contributing

Contributions welcome! This project follows research-grade standards:

**Development Guidelines:**
1. 🧠 **Models**: Add architectures to `src/models/` (no training logic)
2. 📓 **Experiments**: Create notebooks for new research directions  
3. ⚙️ **Config**: Update `configs/config.yaml` for new parameters
4. 🔄 **Reproducibility**: Always use `set_seed(42)` for deterministic results
5. 🧪 **Testing**: Add tests for new functionality
6. 📚 **Documentation**: Update relevant docs and guides

**Contributing Process:**
```bash
# 1. Fork the repository
git clone https://github.com/your-username/AI-Based-approach-for-prioritization-of-genetic-mutations.git

# 2. Create feature branch
git checkout -b feature/amazing-new-feature

# 3. Make changes and test
jupyter notebook  # Test in research pipeline
python app.py     # Test web interface

# 4. Commit and push
git commit -m "Add amazing new feature"
git push origin feature/amazing-new-feature

# 5. Open Pull Request
```

---

## 🙏 Acknowledgments

### Scientific Community
- **ClinVar Database** for providing curated variant classifications
- **gnomAD Consortium** for population frequency data
- **ACMG/AMP Guidelines** for variant interpretation standards
- **Open Grants** and **Medical Research Councils** for funding genomics research

### Technical Stack
- **PyTorch Team** for the excellent deep learning framework
- **Flask** developers for the lightweight web framework
- **Jupyter Project** for interactive research environments
- **scikit-learn** team for robust machine learning tools
- **Matplotlib/Seaborn** for publication-quality visualizations

### Open Source Community
- All GitHub contributors and issue reporters
- Scientific Python ecosystem maintainers
- Medical informatics and bioinformatics communities

---

## 📞 Contact & Support

### 👨‍💻 Main Developer
**Surya Hariharan**

- 🐙 **GitHub**: [@Surya-Hariharan](https://github.com/Surya-Hariharan)
- 💼 **LinkedIn**: [Surya HA](https://linkedin.com/in/surya-ha)
- 📧 **Email**: [suryahariharan2006@gmail.com](mailto:suryahariharan2006@gmail.com)

### 🔗 Project Links
- 🌐 **Repository**: [AI-Based Genetic Mutation Prioritization](https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations)
- 📚 **Documentation**: Complete guides and API documentation
- 🐛 **Issues**: [Report Bugs](https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations/issues)
- 💬 **Discussions**: [Community Forum](https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations/discussions)

---

### 🆘 Getting Help

**Quick Issues:**
- ❓ **Setup Problems**: Check [QUICK_START.md](QUICK_START.md) or [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations/issues)
- 💡 **Feature Requests**: Open an issue with detailed use case
- 🤝 **Collaboration**: Contact via GitHub or email

**Documentation:**
- 📖 **Complete Guide**: This README
- 🚀 **Quick Setup**: [QUICK_START.md](QUICK_START.md) 
- 🌐 **API Docs**: [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)
- 🔬 **Research**: Jupyter notebooks with detailed explanations

### 🏥 Clinical & Research Inquiries

**For clinical applications or research collaborations:**
- Open a GitHub issue with "[CLINICAL]" or "[RESEARCH]" prefix
- Describe your use case, data requirements, and regulatory environment
- We're happy to discuss customization for specific clinical workflows

### 🐛 Bug Reports

**When reporting issues, please include:**
```bash
# System information
python --version
pip list | grep -E "torch|flask|pandas|numpy"
uname -a  # Linux/Mac

# Error details
# - Full error traceback
# - Steps to reproduce
# - Expected vs actual behavior
# - Sample data (if applicable, anonymized)
```

---

## 🎯 Project Impact

### 🧬 Research Applications
- **Genomics Research**: Accelerate variant discovery and interpretation
- **Population Studies**: Large-scale genetic association analysis
- **Functional Genomics**: Prioritize variants for experimental validation
- **Precision Medicine**: Enable personalized treatment strategies

### 🏥 Clinical Potential
- **Diagnostic Support**: Assist clinical geneticists in variant classification
- **Rare Disease**: Help diagnose undiagnosed genetic conditions
- **Cancer Genomics**: Identify oncogenic variants for targeted therapy
- **Pharmacogenomics**: Predict drug response and adverse reactions

### 📊 Technical Contributions
- **Open Science**: Reproducible research pipeline for genomics ML
- **Best Practices**: Template for production-ready ML in healthcare
- **Education**: Learning resource for AI in genomics
- **Community**: Platform for collaborative algorithm development

---

**🧬 Built for Precision Medicine | Research-Grade | Production-Ready | Open Science**

*Advancing genomics through AI-powered variant interpretation and clinical decision support* 🚀
