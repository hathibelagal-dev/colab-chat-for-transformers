# Minimal Transformers CLI Chat Application

[![PyPI version](https://badge.fury.io/py/colab-chat.svg)](https://pypi.org/project/colab-chat/)

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- **Session management**: Save and load chat history automatically.
- **Opt-in Core Tools**: Choose which system tools to enable for the model.
- **Streaming support**: Real-time token output when tools are disabled.

## Prerequisites
- Python 3.8+
- (Optional but recommended) A virtual environment.

## Installation

Install the package directly from PyPI:
```bash
pip install colab-chat
```

Alternatively, for local development:
1. Clone the repository.
2. Install it using pip:
   ```bash
   pip install .
   ```

## Usage

### GGUF Support
You can chat with GGUF models by pointing to a `.gguf` file:
```bash
colab_chat --model path/to/model.gguf
```
Or by specifying the repository and the GGUF file separately:
```bash
colab_chat --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf_file tinyllama-1.1b-chat-v1.0.Q6_K.gguf
```

### Enabling Core Tools
By default, all tools are disabled for security. You can enable them individually using flags:

- **`--calculate`**: Enable mathematical expression evaluation.
- **`--shell`**: Enable execution of bash shell commands.
- **`--read`**: Enable reading files from your system.
- **`--write`**: Enable writing or updating files on your system.
- **`--yolo`**: Enable **all** core tools at once (BE CAREFUL).
- **`--system_prompt`**: Path to a text file containing a custom system prompt.

**Example: Enable all tools**
```bash
colab_chat --yolo
```

> [!WARNING]
> **Security Note**: Enabling `--shell`, `--read`, `--write`, or `--yolo` gives the LLM direct access to your system. Always review the tool calls (indicated by `[*] Calling tool: ...`) in your console.

### Save/Load Sessions
Sessions are saved to the `sessions/` directory in your current working folder.

- **Load a previous session**:
  ```bash
  colab_chat --load sessions/chat_20260407_120000.json
  ```


### Advanced Options
- `--model`: Hugging Face model ID.
- `--max_tokens`: Maximum new tokens to generate (default: 8192).
- `--no_save`: Disable session saving.
- `--session_name`: Set a custom name for the saved session.

## Authors
- **Ashraff Hathibelagal**
- **Gemini CLI** (Co-authored this project)
