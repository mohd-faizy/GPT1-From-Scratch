"""
GPT-1 Model Architecture Implementation.
Reference: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with causal masking.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        
        self.d_head = config.d_model // config.n_head
        self.n_head = config.n_head
        self.d_model = config.d_model
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask buffer (not a parameter)
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Dim)
        Returns:
            Output tensor of shape (Batch, Seq_Len, Dim)
        """
        B, T, C = x.size() # Batch, Time (Seq Len), Channels (Dim)
        
        # Calculate Query, Key, Value
        # [B, T, C] -> [B, T, 3 * C]
        qkv = self.c_attn(x)
        
        # Split into Q, K, V and reshape for multi-head attention
        # [B, T, 3 * C] -> [B, T, 3, n_head, d_head] -> [3, B, n_head, T, d_head]
        q, k, v = qkv.view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        
        # Scaled Dot-Product Attention
        # [B, n_head, T, d_head] @ [B, n_head, d_head, T] -> [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask (mask future tokens)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Aggregate values
        # [B, n_head, T, T] @ [B, n_head, T, d_head] -> [B, n_head, T, d_head]
        y = att @ v
        
        # Reassemble heads
        # [B, n_head, T, d_head] -> [B, T, n_head, d_head] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.act = nn.GELU() # GPT-1 uses GELU
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, T, d_ff] -> [B, T, C]
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer Decoder Block.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    Full GPT-1 Model.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.d_model),
            'wpe': nn.Embedding(config.max_seq_len, config.d_model),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.d_model),
        })
        
        # Language Model Head (tied weights with embedding is common, but GPT-1 paper implies separate or tied? 
        # Usually tied in modern implementations, but we'll stick to a linear layer for clarity or tie it if requested.
        # Original GPT-1 implementation (OpenAI) tied weights. We will tie them.)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Input token indices of shape (Batch, Seq_Len)
            targets: Target token indices of shape (Batch, Seq_Len) (optional)
            
        Returns:
            logits: Output logits of shape (Batch, Seq_Len, Vocab_Size)
            loss: CrossEntropy loss (if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_seq_len}"
        
        # Positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # [1, t]
        
        # Token + Positional Embeddings
        tok_emb = self.transformer.wte(idx) # [b, t, d_model]
        pos_emb = self.transformer.wpe(pos) # [1, t, d_model]
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # Language Model Head
        logits = self.lm_head(x) # [b, t, vocab_size]
        
        loss = None
        if targets is not None:
            # Shift so that we predict the next token
            # But standard practice in forward() is to return logits for all positions.
            # Loss calculation usually handles shifting or we do it here.
            # Let's do standard CrossEntropy on flattened views.
            
            # NOTE: For training, we usually pass inputs as x[:, :-1] and targets as x[:, 1:] outside.
            # But if we pass full sequence and want to train next token prediction:
            # logits will be for positions 0..T-1 predicting 1..T
            
            # Let's assume standard causal LM training where targets are the same sequence shifted?
            # Or caller handles shifting. Let's assume caller handles shifting for flexibility, 
            # OR we compute loss on the fly if targets are provided.
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
