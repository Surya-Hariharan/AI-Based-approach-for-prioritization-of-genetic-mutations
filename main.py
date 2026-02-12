#!/usr/bin/env python3
"""
Main entry point for the Genetic Mutation Prioritization System
Supports both research (Jupyter) and production (Flask web app) workflows
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description='Genetic Mutation Prioritization System')
    parser.add_argument('--mode', choices=['web', 'research', 'train'], 
                       default='web', help='Run mode')
    parser.add_argument('--host', default='127.0.0.1', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', default='development', 
                       choices=['development', 'production', 'testing'],
                       help='Configuration profile')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        print("🧬 Starting Genetic Mutation Prioritization Web Application...")
        print(f"📍 Project Root: {PROJECT_ROOT}")
        print(f"🌐 Server: http://{args.host}:{args.port}")
        print(f"⚙️  Configuration: {args.config}")
        print("=" * 60)
        
        # Import and run Flask app
        os.chdir(PROJECT_ROOT)
        from backend.app import app, load_models
        
        # Load ML models
        models_loaded = load_models()
        if not models_loaded:
            print("\n❌ WARNING: No models found!")
            print("   Please train models first using:")
            print("   python main.py --mode research")
            print("   Or run notebooks: 02_baseline_training.ipynb, 03_mlp_training.ipynb")
        
        # Run Flask application
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.mode == 'research':
        print("🔬 Starting Research Environment...")
        print("📚 Opening Jupyter Notebook for research workflow")
        print("📋 Recommended notebook order:")
        print("   1. 00_data_pipeline.ipynb     # Data processing")
        print("   2. 01_data_exploration.ipynb  # EDA")
        print("   3. 02_baseline_training.ipynb # Baseline model")
        print("   4. 03_mlp_training.ipynb      # Deep learning")
        print("   5. 04_ensemble_training.ipynb # Ensemble")
        print("   6. 05_uncertainty_analysis.ipynb # Uncertainty")
        print("   7. 06_gene_level_ranking.ipynb # Gene ranking")
        
        # Launch Jupyter
        import subprocess
        os.chdir(PROJECT_ROOT / 'notebooks')
        subprocess.run(['jupyter', 'notebook'])
    
    elif args.mode == 'train':
        print("🏋️ Starting Model Training Pipeline...")
        print("This will run the complete training pipeline programmatically")
        
        # Import training modules
        try:
            from src.utils.seed import set_seed
            from src.utils.config import Config
            
            set_seed(42)
            config = Config(str(PROJECT_ROOT / 'configs' / 'config.yaml'))
            
            print("✅ Configuration loaded")
            print("🔄 Training pipeline not yet implemented")
            print("   Please use research mode (jupyter notebooks) for now")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Please ensure all dependencies are installed")

if __name__ == '__main__':
    main()