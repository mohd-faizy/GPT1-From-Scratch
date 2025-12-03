# Contributing to GPT-1 From Scratch

Thank you for your interest in contributing! We welcome contributions from the community to make this project better.

## How to Contribute

1.  **Fork the Repository**: Click the "Fork" button on the top right of the repository page.
2.  **Clone your Fork**:
    ```bash
    git clone https://github.com/yourusername/GPT1-From-Scratch.git
    cd GPT1-From-Scratch
    ```
3.  **Create a Branch**:
    ```bash
    git checkout -b feature/amazing-feature
    ```
4.  **Make Changes**: Implement your feature or fix.
5.  **Run Tests**: Ensure all tests pass.
    ```bash
    uv run tests/test_model.py
    ```
6.  **Commit Changes**:
    ```bash
    git commit -m "Add amazing feature"
    ```
7.  **Push to Branch**:
    ```bash
    git push origin feature/amazing-feature
    ```
8.  **Open a Pull Request**: Go to the original repository and click "New Pull Request".

## Development Guidelines

-   **Code Style**: We follow [PEP 8](https://peps.python.org/pep-0008/). Please ensure your code is formatted correctly.
-   **Type Hinting**: All new functions and classes must have Python type hints.
-   **Docstrings**: Use Google-style docstrings for all public modules, classes, and functions.
-   **Testing**: Add unit tests for any new logic in `tests/`.

## Project Structure

-   `src/`: Core library code.
-   `scripts/`: Executable scripts.
-   `tests/`: Unit tests.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue.
