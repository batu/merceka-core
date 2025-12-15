# merceka_core

Core utilities for merceka projects.

## Installation

```bash
pip install merceka_core
```

## Usage

```python
from merceka_core.llm import LLM

# Local Ollama model
llm = LLM("gemma3:27b")
response = llm.generate("Hello, how are you?")

# Cloud model via OpenRouter
llm = LLM("openrouter/google/gemini-2.5-flash-lite-preview-09-2025")
response = llm.generate("Hello, how are you?")
```

## Development

This project uses [nbdev](https://nbdev.fast.ai/) for literate programming.

```bash
# Install dependencies
uv sync

# Start Jupyter
uv run jupyter lab

# Export notebooks to Python modules
uv run nbdev_export
```

