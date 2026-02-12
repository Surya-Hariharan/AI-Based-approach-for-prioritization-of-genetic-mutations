"""
Flask Web Application for Genetic Mutation Prioritization
Provides REST API and web interface for mutation pathogenicity prediction
"""

import os
import sys
import json
import torch
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.models.baseline import LogisticRegression as BaselineModel
    from src.models.mlp import MLP
    from src.preprocessing.preprocessing import load_preprocessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run this from the project root or install the package")
    sys.exit(1)

# Initialize Flask app with proper paths
app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, 'frontend', 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'frontend', 'static'))

# Enable CORS for frontend-backend separation
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'data', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'vcf', 'txt'}
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'genetic-mutation-ai-secret-key-2026')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model storage
models = {}
preprocessor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load all available trained models"""
    global models, preprocessor
    
    print("Loading models...")
    
    # Load preprocessor
    preprocessor_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            print("✓ Preprocessor loaded")
        except Exception as e:
            print(f"⚠ Failed to load preprocessor: {e}")
    
    # Load baseline model
    baseline_path = os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'baseline_model.pth')
    if os.path.exists(baseline_path):
        try:
            checkpoint = torch.load(baseline_path, map_location=device)
            input_dim = checkpoint.get('input_dim', 50)
            baseline = BaselineModel(input_dim=input_dim).to(device)
            baseline.load_state_dict(checkpoint['model_state_dict'])
            baseline.eval()
            models['baseline'] = baseline
            print("✓ Baseline model loaded")
        except Exception as e:
            print(f"⚠ Failed to load baseline model: {e}")
    
    # Load MLP model
    mlp_path = os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'mlp_best.pth')
    if os.path.exists(mlp_path):
        try:
            checkpoint = torch.load(mlp_path, map_location=device)
            input_dim = checkpoint.get('input_dim', 50)
            hidden_layers = checkpoint.get('hidden_layers', [256, 128, 64])
            mlp = MLP(input_dim=input_dim, hidden_layers=hidden_layers).to(device)
            mlp.load_state_dict(checkpoint['model_state_dict'])
            mlp.eval()
            models['mlp'] = mlp
            print("✓ MLP model loaded")
        except Exception as e:
            print(f"⚠ Failed to load MLP model: {e}")
    
    # Load ensemble model
    ensemble_path = os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'ensemble_model.joblib')
    if os.path.exists(ensemble_path):
        try:
            ensemble = joblib.load(ensemble_path)
            models['ensemble'] = ensemble
            print("✓ Ensemble model loaded")
        except Exception as e:
            print(f"⚠ Failed to load ensemble model: {e}")
    
    if not models:
        print("⚠ Warning: No models found. Please train models first.")
    
    return len(models) > 0


def predict_mutation(features, model_name='mlp'):
    """
    Predict pathogenicity for a single mutation
    
    Args:
        features: numpy array or pandas DataFrame of mutation features
        model_name: which model to use ('baseline', 'mlp', 'ensemble')
    
    Returns:
        dict with prediction, probability, and confidence
    """
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not available. Available: {list(models.keys())}")
    
    model = models[model_name]
    
    # Convert to numpy if needed
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    # Ensure 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Make prediction
    if model_name == 'ensemble':
        # Sklearn model
        proba = model.predict_proba(features)[:, 1]
        pred = model.predict(features)
    else:
        # PyTorch model
        with torch.no_grad():
            X = torch.FloatTensor(features).to(device)
            output = model(X)
            proba = torch.sigmoid(output).cpu().numpy().flatten()
            pred = (proba > 0.5).astype(int)
    
    results = []
    for i in range(len(proba)):
        prob = float(proba[i])
        prediction = int(pred[i])
        
        results.append({
            'prediction': 'Pathogenic' if prediction == 1 else 'Benign',
            'probability': prob,
            'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low',
            'pathogenic_score': prob,
            'benign_score': 1 - prob
        })
    
    return results[0] if len(results) == 1 else results


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html', models=list(models.keys()))


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'device': str(device)
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models"""
    model_info = {}
    for name, model in models.items():
        if hasattr(model, 'parameters'):
            # PyTorch model
            params = sum(p.numel() for p in model.parameters())
            model_info[name] = {
                'type': 'neural_network',
                'parameters': params,
                'device': str(next(model.parameters()).device)
            }
        else:
            # Sklearn model
            model_info[name] = {
                'type': 'ensemble',
                'estimators': len(model.estimators_) if hasattr(model, 'estimators_') else 'N/A'
            }
    
    return jsonify(model_info)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict mutation pathogenicity
    Accepts JSON with features or file upload
    """
    try:
        model_name = request.form.get('model', 'mlp')
        
        if model_name not in models:
            return jsonify({
                'error': f"Model '{model_name}' not available",
                'available_models': list(models.keys())
            }), 400
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read CSV file
                df = pd.read_csv(filepath)
                
                # Assume all columns are features (or remove known non-feature columns)
                exclude_cols = ['mutation_id', 'gene', 'label', 'pathogenicity']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                features = df[feature_cols].values
                
                # Make predictions
                results = predict_mutation(features, model_name)
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'model': model_name,
                    'predictions': results if isinstance(results, list) else [results],
                    'count': len(results) if isinstance(results, list) else 1
                })
        
        # Handle JSON data
        elif request.is_json:
            data = request.get_json()
            
            if 'features' not in data:
                return jsonify({'error': 'Missing "features" field in JSON'}), 400
            
            features = np.array(data['features'])
            result = predict_mutation(features, model_name)
            
            return jsonify({
                'success': True,
                'model': model_name,
                'prediction': result
            })
        
        else:
            return jsonify({'error': 'No file or JSON data provided'}), 400
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict multiple mutations at once"""
    try:
        data = request.get_json()
        model_name = data.get('model', 'mlp')
        
        if 'mutations' not in data:
            return jsonify({'error': 'Missing "mutations" field'}), 400
        
        mutations = np.array(data['mutations'])
        results = predict_mutation(mutations, model_name)
        
        return jsonify({
            'success': True,
            'model': model_name,
            'predictions': results if isinstance(results, list) else [results],
            'count': len(results) if isinstance(results, list) else 1
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    try:
        ranked_genes_path = os.path.join(PROJECT_ROOT, 'reports', 'results', 'ranked_genes.csv')
        
        stats = {
            'models_available': len(models),
            'models': list(models.keys())
        }
        
        if os.path.exists(ranked_genes_path):
            df = pd.read_csv(ranked_genes_path)
            stats['total_genes_ranked'] = len(df)
            stats['top_genes'] = df.head(10)[['gene', 'mean_score']].to_dict('records') if 'gene' in df.columns else []
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print(f"\n{'='*60}")
        print("🧬 Genetic Mutation Prioritization Server")
        print(f"{'='*60}")
        print(f"Models loaded: {', '.join(models.keys())}")
        print(f"Device: {device}")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Frontend path: {app.template_folder}")
        print(f"Static path: {app.static_folder}")
        print(f"{'='*60}\n")
        
        # Run the app
        debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
        port = int(os.getenv('PORT', 5000))
        host = os.getenv('HOST', '127.0.0.1')
        
        print(f"Starting server on http://{host}:{port}")
        app.run(debug=debug_mode, host=host, port=port)
    else:
        print("\n❌ Error: No models found. Please train models first using the notebooks.")
        print("Run notebooks 02_baseline_training.ipynb, 03_mlp_training.ipynb, or 04_ensemble_training.ipynb")
        print(f"Expected model paths:")
        print(f"  - {os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'baseline_model.pth')}")
        print(f"  - {os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'mlp_best.pth')}")
        print(f"  - {os.path.join(PROJECT_ROOT, 'reports', 'results', 'checkpoints', 'ensemble_model.joblib')}")
        sys.exit(1)
