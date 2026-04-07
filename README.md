# Minimal Transformers CLI Chat Application

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- **Session management**: Save and load chat history automatically.

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
Start with the default model (`SmolLM2-135M-Instruct`):
```bash
python chat.py
```

### Save/Load Sessions
By default, sessions are saved to the `sessions/` directory with a timestamped filename (e.g., `sessions/chat_20260407_120000.json`).

- **Custom session name**:
  ```bash
  python chat.py --session_name my_cool_chat
  ```
- **Load a previous session**:
  ```bash
  python chat.py --load sessions/my_cool_chat.json
  ```
- **Disable saving**:
  ```bash
  python chat.py --no_save
  ```

### Advanced Options
- `--model`: Hugging Face model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`).
- `--max_tokens`: Maximum new tokens to generate (default: 500).

## Example
```
Loading model: HuggingFaceTB/SmolLM2-135M-Instruct...
Chat started! Type 'exit' or 'quit' to end the conversation.
This session will be saved to: sessions/chat_20260407_123456.json
--------------------------------------------------
You: Hello!
Assistant: Hi there! How can I help you?
```
