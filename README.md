# Minimal Transformers CLI Chat Application

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- **Session management**: Save and load chat history automatically.
- **Opt-in Core Tools**: Choose which system tools to enable for the model.

## Prerequisites
- Python 3.8+
- (Optional but recommended) A virtual environment.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Chat
Start with the default model (`SmolLM2-1.7B-Instruct`) and **no tools enabled**:
```bash
python chat.py
```

### Enabling Core Tools
By default, all tools are disabled for security. You can enable them individually using flags:

- **`--calculate`**: Enable mathematical expression evaluation.
- **`--shell`**: Enable execution of bash shell commands.
- **`--read`**: Enable reading files from your system.
- **`--write`**: Enable writing or updating files on your system.
- **`--yolo`**: Enable **all** core tools at once (BE CAREFUL).

**Example: Enable all tools**
```bash
python chat.py --yolo
```

> [!WARNING]
> **Security Note**: Enabling `--shell`, `--read`, `--write`, or `--yolo` gives the LLM direct access to your system. Always review the tool calls (indicated by `[*] Calling tool: ...`) in your console.

### Save/Load Sessions
Sessions are saved to the `sessions/` directory.

- **Load a previous session**:
  ```bash
  python chat.py --load sessions/chat_20260407_120000.json
  ```

### Advanced Options
- `--model`: Hugging Face model ID.
- `--max_tokens`: Maximum new tokens to generate (default: 500).
- `--no_save`: Disable session saving.
- `--session_name`: Set a custom name for the saved session.
