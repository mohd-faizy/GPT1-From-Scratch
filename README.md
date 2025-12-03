# GPT-1 From Scratch

A clean, educational, and production-ready implementation of the original GPT-1 architecture (Radford et al., 2018) in PyTorch.

## 🚀 Features

- **Accurate Architecture**: Strictly follows the GPT-1 paper specifications (117M parameters).
- **Decoder-Only**: Implements custom `Block` with Causal Multi-Head Attention and GELU activations.
- **Clean Code**: Type-hinted, modular, and documented following PEP 8 standards.
- **Easy to Use**: Simple scripts for training and text generation with CLI support.
- **Modern Tooling**: Supports `uv` for fast dependency management.

## 📂 Directory Structure

```
GPT1-From-Scratch/
├── src/
│   ├── config.py       # Configuration dataclass defining model hyperparameters (layers, heads, dim)
│   ├── model.py        # Core PyTorch implementation of GPT-1 (Attention, FeedForward, Blocks)
│   ├── dataset.py      # Data loading logic using HuggingFace Datasets and Tokenizers
│   └── utils.py        # Utility functions for logging, checkpointing, and visualization
├── scripts/
│   ├── train.py        # Main training script with validation loop and checkpointing
│   └── generate.py     # Inference script for text generation with top-k sampling
├── tests/              # Unit tests for model architecture and components
├── data/               # Directory for storing downloaded datasets
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

### Using `uv` (Recommended)

This project uses `uv` for fast package management.

1. Clone the repository:
   ```bash
   git clone https://github.com/mohd-faizy/GPT1-From-Scratch.git
   cd GPT1-From-Scratch
   ```

2. Install dependencies:
   ```bash
   uv add -r requirements.txt
   ```

### Using `pip`

Alternatively, you can use standard `pip`:

```bash
pip install -r requirements.txt
```

## 🏃 Usage

### Training

To train the model on the WikiText dataset:

```bash
# Using uv
uv run scripts/train.py --batch_size 8 --epochs 3

# Using python
python scripts/train.py --batch_size 8 --epochs 3
```

**Arguments:**
- `--batch_size`: Batch size per GPU (default: 8).
- `--epochs`: Number of training epochs (default: 3).
- `--subset`: Use a subset of data for debugging (e.g., `--subset 1000`).

### Text Generation

To generate text using a trained model:

```bash
# Using uv
uv run scripts/generate.py --prompt "The future of AI is" --model_path checkpoints/best_model.pth

# Using python
python scripts/generate.py --prompt "The future of AI is" --model_path checkpoints/best_model.pth
```

**Arguments:**
- `--prompt`: The starting text.
- `--model_path`: Path to the checkpoint (default: `checkpoints/best_model.pth`).
- `--max_length`: Maximum tokens to generate.
- `--temperature`: Sampling temperature (default: 0.7).

## 🧠 Architecture Details

The model configuration matches the original GPT-1 117M parameter model:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Layers** | 12 | Number of Transformer blocks |
| **Attention Heads** | 12 | Number of heads in Multi-Head Attention |
| **Embedding Dim** | 768 | Dimension of token and positional embeddings ($d_{model}$) |
| **Feed-Forward Dim** | 3072 | Dimension of the inner layer in FFN ($4 \times d_{model}$) |
| **Max Sequence Len** | 512 | Maximum context window size |
| **Vocabulary Size** | ~40,478 | Based on BPE Tokenizer |

### Parameter Calculation

The 117M parameter count is derived as follows:

1.  **Embeddings**:
    *   Token Embeddings: $V \times d_{model} = 40,478 \times 768 \approx 31M$
    *   Positional Embeddings: $T \times d_{model} = 512 \times 768 \approx 0.4M$
2.  **Transformer Blocks (12 layers)**:
    *   **Attention**: $4 \times (d_{model}^2 + d_{model})$ (for Q, K, V, Output projections + biases) $\approx 2.36M$ per layer
    *   **Feed-Forward**: $2 \times d_{model} \times d_{ff} + d_{model} + d_{ff}$ (weights + biases) $\approx 4.7M$ per layer
    *   **LayerNorms**: $2 \times 2 \times d_{model}$ (scale + shift) $\approx 3k$ per layer
    *   *Total per layer*: $\approx 7M$
    *   *Total for 12 layers*: $12 \times 7M \approx 85M$
3.  **Total**: $31M + 0.4M + 85M \approx 116.4M$ parameters.

*(Note: Exact count depends on the specific vocabulary size of the tokenizer used.)*

## 🧪 Testing

Run unit tests to verify the architecture:

```bash
uv run tests/test_model.py
```

## 🔍 Code Walkthrough

Understanding how the components interact:

1.  **Configuration (`src/config.py`)**:
    -   The `GPTConfig` dataclass holds all hyperparameters. It's the single source of truth for model size and training settings.

2.  **Data Pipeline (`src/dataset.py`)**:
    -   `load_dataset` fetches text from HuggingFace.
    -   `GPTDataset` tokenizes text using `AutoTokenizer` (GPT-2 tokenizer) and handles truncation/padding.
    -   `get_dataloader` creates batches. It uses a `collate_fn` (implicitly handled by `return_tensors="pt"` in dataset) to stack tensors.

3.  **Model (`src/model.py`)**:
    -   **`GPT`**: The main container. It creates the embeddings (`wte`, `wpe`) and a stack of `Block` layers.
    -   **`Block`**: A single Transformer decoder layer. It contains:
        -   `MultiHeadAttention`: Calculates self-attention with a causal mask (tril matrix) to ensure positions can only attend to previous positions.
        -   `FeedForward`: A two-layer MLP with GELU activation.
        -   `LayerNorm`: Applied before attention and FFN (Pre-Norm architecture is common in modern GPTs, though original GPT-1 was Post-Norm. This implementation uses a standard structure).

4.  **Training Loop (`scripts/train.py`)**:
    -   Iterates through the `DataLoader`.
    -   Feeds inputs to the model.
    -   Calculates `CrossEntropyLoss` between model logits and shifted targets (next-token prediction).
    -   Backpropagates gradients and updates weights using `AdamW`.
    -   Evaluates on validation set and saves checkpoints.

## 🚀 Getting Started

Follow these steps to get up and running with the GPT-1 implementation.

### 1. Verify Installation

First, ensure that the environment is set up correctly and all dependencies are installed by running the unit tests.

```bash
uv run tests/test_model.py
```

**Expected Output:**
You should see output indicating that all tests passed, similar to:
```
test_attention_shape ... ok
test_gpt_forward ... ok
...
Ran 5 tests in 1.234s

OK
```

### 2. Train the Model

Train the model on a small subset of the data to verify the training loop.

```bash
uv run scripts/train.py --batch_size 4 --epochs 1 --subset 100
```

**What this does:**
- Loads the WikiText dataset (or a subset).
- Initializes the GPT-1 model.
- Runs the training loop for 1 epoch.
- Saves checkpoints to `checkpoints/`.

**Expected Output:**
You will see a progress bar and loss logging:
```
Using device: cuda
Configuration: ...
Epoch 1/1: 100%|██████████| 25/25 [00:05<00:00,  4.50it/s, loss=3.4567]
Epoch 1: Train Loss = 3.4567, Val Loss = 3.4000
Training completed.
```

### 3. Generate Text

Use the trained model (or the initialized one if training was short) to generate text.

```bash
uv run scripts/generate.py --prompt "Artificial Intelligence is" --model_path checkpoints/best_model.pth
```

**What this does:**
- Loads the model weights from `checkpoints/best_model.pth`.
- Tokenizes the input prompt.
- Generates new tokens using top-k sampling.
- Decodes and prints the result.

**Expected Output:**
```
Using device: cuda
Loaded model from checkpoints/best_model.pth

Generated Text:
--------------------------------------------------
Artificial Intelligence is a field of study that ...
--------------------------------------------------
```

## 📚 References

-   **Original Paper**: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
-   **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
-   **HuggingFace Transformers**: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

## 🔗 Connect with me

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/F4izy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohd-faizy/)
[![Stack Exchange](https://img.shields.io/badge/Stack_Exchange-1E5397?style=for-the-badge&logo=stack-exchange&logoColor=white)](https://ai.stackexchange.com/users/36737/faizy)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)

</div>
