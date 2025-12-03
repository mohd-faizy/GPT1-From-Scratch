"""
Utility functions for logging, checkpointing, and visualization.
"""

import os
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict

def setup_logging(log_file: str = "training.log") -> None:
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    path: str = "checkpoints/checkpoint.pth"
) -> None:
    """
    Saves a training checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(
    path: str, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Loads a training checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    logging.info(f"Checkpoint loaded from {path} (Epoch {checkpoint.get('epoch', 'Unknown')})")
    return checkpoint

def plot_loss(losses: List[float], save_path: str = "training_loss.png", window: int = 50) -> None:
    """
    Plots and saves the training loss curve.
    """
    if not losses:
        return

    plt.figure(figsize=(10, 5))
    
    # Plot raw loss
    plt.plot(losses, alpha=0.3, label="Raw Loss")
    
    # Plot smoothed loss
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smoothed, label=f"Smoothed (window={window})")
    
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss plot saved to {save_path}")
