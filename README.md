# Minimal Transformers CLI Chat Application

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- **Session management**: Save and load chat history automatically.
- **Core Tool Use**: Built-in tools that allow the model to interact with your system.

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
Start with the default model (`SmolLM2-1.7B-Instruct`):
```bash
python chat.py
```

### Core Tools
The application provides the following powerful built-in tools:
- `run_shell_command`: Execute any bash command and see the output.
- `read_file`: Read the content of any file on your system.
- `write_file`: Create or overwrite files with specific content.
- `calculate`: Evaluate mathematical expressions.

**Example queries:**
- *"What files are in my current directory?"*
- *"Read the content of requirements.txt"*
- *"Create a script named hello.py that prints 'Hello World'"*
- *"What is 1234 * 5678?"*

> [!WARNING]
> **Security Note**: These tools give the LLM direct access to your shell and file system. Always review the tool calls (indicated by `[*] Calling tool: ...`) before they execute.

### Save/Load Sessions
Sessions are saved to the `sessions/` directory.

- **Load a previous session**:
  ```bash
  python chat.py --load sessions/chat_20260407_120000.json
  ```
- **Custom session name**:
  ```bash
  python chat.py --session_name my_session
  ```

### Advanced Options
- `--model`: Hugging Face model ID (default: `HuggingFaceTB/SmolLM2-1.7B-Instruct`).
- `--no_tools`: Disable all tool-use functionality.
- `--max_tokens`: Maximum new tokens to generate (default: 500).
