"""
Training script for GPT-1.
"""

import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPTConfig
from src.model import GPT
from src.dataset import get_dataloader
from src.utils import setup_logging, save_checkpoint, plot_loss

def get_lr_scheduler(optimizer, warmup_steps):
    """Create learning rate scheduler with warmup"""
    return LambdaLR(optimizer, lambda step: min(step/warmup_steps, 1.0))

def train(args):
    # Setup logging
    setup_logging()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Config
    config = GPTConfig()
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.epochs = args.epochs
    if args.subset: config.subset_size = args.subset
    
    logging.info(f"Configuration: {config}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = len(tokenizer)
    
    # DataLoaders
    train_loader = get_dataloader(config, tokenizer, split="train")
    # Try validation split, fallback to train subset if not available or for simple testing
    try:
        val_loader = get_dataloader(config, tokenizer, split="validation")
    except:
        logging.warning("Validation split not found or failed. Using a subset of training data for validation.")
        val_loader = get_dataloader(config, tokenizer, split="train") # In real scenario, split manually
        
    # Model
    model = GPT(config).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps)
    
    # Training Loop
    best_val_loss = float('inf')
    all_losses = []
    
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in pbar:
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # Forward pass
            # GPT-1 is autoregressive. We predict the next token.
            # Inputs: [B, T]
            # Targets: [B, T] shifted by 1
            
            # In our model.forward, we handle simple forward. 
            # Let's prepare targets here.
            # If we pass inputs as is, we need to handle shifting for loss.
            
            # Standard causal LM training:
            # Input: x_0, x_1, ..., x_{T-1}
            # Target: x_1, x_2, ..., x_T
            
            # However, our dataset returns fixed length sequences.
            # Let's use the full sequence and mask the last token for input and first for target.
            
            # Input:  [A, B, C, D] -> predict [B, C, D, E]
            # But we only have [A, B, C, D].
            # So Input: [A, B, C], Target: [B, C, D]
            
            # Actually, standard way with fixed block size:
            # Input: [A, B, C, D]
            # Target: [A, B, C, D]
            # Model outputs logits for positions.
            # Loss calculated on shifted views.
            
            outputs, _ = model(inputs) # [B, T, V]
            
            # Shift for loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            all_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                outputs, _ = model(inputs)
                
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, "checkpoints/best_model.pth")
        
        model.train()
        
    # Save final model
    save_checkpoint(model, optimizer, config.epochs, avg_train_loss, "checkpoints/final_model.pth")
    plot_loss(all_losses)
    logging.info("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-1 Model")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--subset", type=int, help="Use subset of data for debugging")
    
    args = parser.parse_args()
    train(args)
