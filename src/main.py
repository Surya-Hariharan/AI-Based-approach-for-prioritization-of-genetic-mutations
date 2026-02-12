import argparse
import sys
import os

# Set environment variable to allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.prioritize import main as prioritize_main
from src.train_ensemble import main as train_ensemble_main

def main():
    parser = argparse.ArgumentParser(description="Genetic Mutation Prioritization System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train a new model", add_help=False)
    
    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model", add_help=False)
    
    # Prioritize
    prioritize_parser = subparsers.add_parser("prioritize", help="Prioritize mutations in new data", add_help=False)
    
    # Train Ensemble
    ensemble_parser = subparsers.add_parser("train-ensemble", help="Train ensemble model", add_help=False)
    # We could forward arguments, but each script handles its own args through argparse.
    # A cleaner way is to let them handle sys.argv, but that requires patching.
    # Or we just import their main functionality if refactored.
    # Since they use argparse inside main(), we can use `sys.argv` manipulation or 
    # refactor them to accept args. 
    # For now, let's keep it simple: just a dispatcher that calls the script via subprocess?
    # No, importing is better. But we need to handle arguments.
    
    # Actually, the best way for a "research grade" main.py is to just be a CLI entry point 
    # that dispatches to the modules.
    
    args, unknown = parser.parse_known_args()
    
    # Forward unknown args to the specific module
    sys.argv = [sys.argv[0]] + unknown
    
    if args.command == "train":
        train_main()
    elif args.command == "evaluate":
        evaluate_main()
    elif args.command == "prioritize":
        prioritize_main()
    elif args.command == "train-ensemble":
        train_ensemble_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
