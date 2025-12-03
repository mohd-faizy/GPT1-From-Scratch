"""
Dataset and DataLoader implementation for GPT-1 training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional
from .config import GPTConfig

class GPTDataset(Dataset):
    """
    Custom Dataset for GPT-1 training.
    """
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int):
        """
        Args:
            texts: List of text samples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension added by return_tensors="pt"
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def get_dataloader(
    config: GPTConfig, 
    tokenizer: PreTrainedTokenizer, 
    split: str = "train"
) -> DataLoader:
    """
    Creates a DataLoader for the specified dataset and split.
    
    Args:
        config: Configuration object.
        tokenizer: Tokenizer instance.
        split: Dataset split ('train', 'validation', 'test').
        
    Returns:
        DataLoader instance.
    """
    # Load dataset
    # Using wikitext-103-raw-v1 as per config, or fallback to wikitext-2 if 103 is too large for quick testing
    # But let's stick to config.
    
    # Handle subset for debugging
    split_name = split
    if config.subset_size is not None and split == "train":
        split_name = f"{split}[:{config.subset_size}]"
    
    try:
        dataset = load_dataset(config.dataset_name, config.dataset_config, split=split_name)
    except Exception as e:
        print(f"Error loading dataset {config.dataset_name}/{config.dataset_config}: {e}")
        # Fallback to wikitext-2 if 103 fails or is not found (though it should be standard)
        print("Falling back to wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_name)

    # Filter empty texts
    texts = [text for text in dataset["text"] if text.strip()]
    
    if not texts:
        raise ValueError(f"No valid texts found in {split} split!")

    gpt_dataset = GPTDataset(texts, tokenizer, config.max_seq_len)
    
    return DataLoader(
        gpt_dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=0, # Windows often has issues with num_workers > 0 in simple scripts
        pin_memory=True
    )
