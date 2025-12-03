"""
Text generation script.
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPTConfig
from src.model import GPT

def generate(
    model: torch.nn.Module, 
    tokenizer: AutoTokenizer, 
    prompt: str, 
    max_length: int = 50, 
    temperature: float = 0.7, 
    top_k: int = 40, 
    device: str = "cpu"
) -> str:
    """
    Generates text from a prompt.
    """
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            # Truncate if sequence is too long
            if tokens.size(1) > model.config.max_seq_len:
                input_tokens = tokens[:, -model.config.max_seq_len:]
            else:
                input_tokens = tokens
                
            logits, _ = model(input_tokens)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_index)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Config and Model
    # Note: In a real scenario, we should load config from checkpoint or saved config file.
    # Here we assume default config matches trained model.
    config = GPTConfig()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)
    
    model = GPT(config).to(device)
    
    # Load weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}, using random weights.")
        
    # Generate
    output = generate(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature, 
        device=device
    )
    
    print("\nGenerated Text:\n" + "-"*50)
    print(output)
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with GPT-1")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    main(args)
