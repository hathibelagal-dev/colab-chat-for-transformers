# Minimal Transformers CLI Chat Application

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- **Session management**: Save and load chat history automatically.
- **Tool Use (Function Calling)**: Support for models that can call external functions.

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

### Tool Use
The application includes built-in tools like `get_weather` and `calculate`.

#### 1. Python Tool Definitions
Create a Python file and define functions with Google-style docstrings.
```bash
python chat.py --tools_file my_tools.py
```

#### 2. JSON Tool Definitions
You can also pass tool schemas via a JSON file.
```bash
python chat.py --tools_json tools.json
```
**Example `tools.json`**:
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_stock_price",
      "description": "Get current stock price",
      "parameters": {
        "type": "object",
        "properties": {
          "symbol": {"type": "string"}
        },
        "required": ["symbol"]
      }
    }
  }
]
```
*Note: If the tool name in the JSON matches a built-in function (e.g., `get_weather`), it will use that function. Otherwise, it will provide a mock response for testing.*

### Save/Load Sessions
Sessions are saved to the `sessions/` directory.

- **Load a previous session**:
  ```bash
  python chat.py --load sessions/chat_20260407_120000.json
  ```

### Advanced Options
- `--model`: Hugging Face model ID.
- `--no_tools`: Disable tool-use functionality.
- `--max_tokens`: Maximum new tokens to generate (default: 500).
