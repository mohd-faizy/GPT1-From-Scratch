"""
Configuration module for GPT-1 model and training parameters.
"""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Configuration class for GPT-1 model architecture and training settings.
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        max_seq_len (int): Maximum sequence length.
        n_layer (int): Number of transformer blocks.
        n_head (int): Number of attention heads.
        d_model (int): Embedding dimension.
        d_ff (int): Dimension of the feed-forward network.
        dropout (float): Dropout probability.
        
        batch_size (int): Training batch size.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for regularization.
        epochs (int): Number of training epochs.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        grad_clip (float): Gradient clipping threshold.
        
        dataset_name (str): Name of the dataset to use.
        dataset_config (str): Dataset configuration/subset.
        subset_size (int): Number of samples to use (for debugging). None for full dataset.
    """
    # Model Architecture (GPT-1 117M parameters)
    vocab_size: int = 40478  # Approximate, will be set by tokenizer
    max_seq_len: int = 512
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    
    # Training Parameters
    batch_size: int = 8  # Adjust based on VRAM
    lr: float = 2.5e-4
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    
    # Dataset Parameters
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1" # Larger dataset for real training
    subset_size: int = None # Set to int for debugging, e.g., 1000

    # Generation Parameters
    temperature: float = 0.7
    top_k: int = 40
