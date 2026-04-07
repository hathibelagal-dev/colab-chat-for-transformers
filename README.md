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
Example question: *"What is the weather in Paris?"* or *"What is 15 * 24?"*.

- **Custom Tools**: Create a Python file (e.g., `my_tools.py`) and define functions with Google-style docstrings.
  ```python
  def get_stock_price(symbol: str):
      """
      Get the current stock price for a given symbol.
      Args:
          symbol: The stock symbol (e.g. AAPL, TSLA).
      """
      return {"symbol": symbol, "price": 150.00}
  ```
  Run with:
  ```bash
  python chat.py --tools_file my_tools.py
  ```

### Save/Load Sessions
Sessions are saved to the `sessions/` directory.

- **Load a previous session**:
  ```bash
  python chat.py --load sessions/chat_20260407_120000.json
  ```
- **Custom session name**:
  ```bash
  python chat.py --session_name research_session
  ```

### Advanced Options
- `--model`: Hugging Face model ID (e.g., `Qwen/Qwen2.5-7B-Instruct`).
- `--no_tools`: Disable tool-use functionality.
- `--max_tokens`: Maximum new tokens to generate (default: 500).

## Example
```
Loading model: HuggingFaceTB/SmolLM2-1.7B-Instruct...
Tools enabled: ['get_weather', 'calculate']
--------------------------------------------------
You: What's the weather in Tokyo?
[*] Calling tool: get_weather({'location': 'Tokyo'})
Assistant: The current weather in Tokyo is sunny with a temperature of 22°C.
```
