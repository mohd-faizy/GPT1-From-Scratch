"""
Unit tests for GPT-1 model.
"""

import sys
import os
import torch
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPTConfig
from src.model import GPT

class TestGPTModel(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(
            vocab_size=1000,
            max_seq_len=128,
            n_layer=2,
            n_head=2,
            d_model=64,
            d_ff=256
        )
        self.model = GPT(self.config)

    def test_output_shape(self):
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits, loss = self.model(x)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_with_targets(self):
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits, loss = self.model(x, targets)
        
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))

    def test_causal_masking(self):
        # Ensure that changing future tokens does not affect past predictions
        batch_size = 1
        seq_len = 10
        x1 = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        x2 = x1.clone()
        x2[0, -1] = (x2[0, -1] + 1) % self.config.vocab_size # Change last token
        
        logits1, _ = self.model(x1)
        logits2, _ = self.model(x2)
        
        # Predictions for positions 0..T-2 should be identical
        # (Prediction at T-2 depends on 0..T-2. Token at T-1 is input for T-1 prediction)
        # Wait, logits at index i corresponds to prediction for token i+1 (usually) or i?
        # In our implementation:
        # logits[i] comes from processing x[0..i].
        # So logits[i] depends on x[0..i].
        # If we change x[T-1], then logits[T-1] changes.
        # But logits[0..T-2] should NOT change.
        
        self.assertTrue(torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :]))
        self.assertFalse(torch.allclose(logits1[:, -1, :], logits2[:, -1, :]))

if __name__ == '__main__':
    unittest.main()
