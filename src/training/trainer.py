import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, Any, Optional, Tuple
from src.config.data_config import Config
from src.evaluation.eval_metrics import calculate_metrics

class Trainer:
    """
    Trainer class for training and evaluating models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
        
    def train(self):
        """
        Runs the full training loop.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.training.get('early_stopping_patience', 5)
        epochs = self.config.training['epochs']
        checkpoint_dir = self.config.training['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train step
            train_loss = self._train_epoch()
            
            # Validation step
            val_loss, val_metrics = self._validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(os.path.join(checkpoint_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save final model and history
        self._save_checkpoint(os.path.join(checkpoint_dir, "final_model.pth"))
        self._save_history()
        return self.history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # Apply sigmoid to convert logits to probabilities for metrics
                probs = torch.sigmoid(outputs)
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
        
        return avg_loss, metrics

    def _save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def _save_history(self):
        log_dir = self.config.training['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=4)
