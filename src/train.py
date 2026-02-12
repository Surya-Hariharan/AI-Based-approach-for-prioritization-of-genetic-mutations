import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.config.data_config import Config
from src.utils import get_data_loaders
from src.models import LogisticRegression, MLP
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Genetic Mutation Prioritization Model")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["baseline", "mlp"], help="Model type to train")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Config
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Override config with CLI args
    if args.model:
        config.config['model']['type'] = args.model
    if args.epochs:
        config.config['training']['epochs'] = args.epochs
    if args.lr:
        config.config['training']['learning_rate'] = args.lr

    print(f"Configuration loaded. Model: {config.model['type']}, Epochs: {config.training['epochs']}, LR: {config.training['learning_rate']}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    try:
        train_loader, val_loader, test_loader, input_dim = get_data_loaders(config)
        print(f"Data loaders created. Input dimension: {input_dim}")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {config.data.get('processed_path')}")
        print("Please ensure data processing pipeline is run before training.")
        return
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    # Model Initialization
    model_type = config.model['type']
    if model_type == "baseline":
        model = LogisticRegression(input_dim)
    elif model_type == "mlp":
        hidden_layers = config.model['mlp']['hidden_layers']
        dropout = config.model['mlp']['dropout']
        model = MLP(input_dim, hidden_layers, dropout)
    else:
        print(f"Unknown model type: {model_type}")
        return

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=config.training['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()
