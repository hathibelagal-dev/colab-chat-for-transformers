# Minimal Transformers CLI Chat Application

A simple command-line interface to chat with any instruction-tuned LLM from the Hugging Face Hub using the `transformers` library.

## Features
- Interactive CLI chat.
- Supports any model with a chat template.
- Automatic device mapping (CPU/GPU).
- Keeps conversation history.

## Prerequisites
- Python 3.8+
- (Optional but recommended) A virtual environment.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the chat application with the default model (`SmolLM2-135M-Instruct`):
```bash
python chat.py
```

Chat with a different model:
```bash
python chat.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Options
- `--model`: Hugging Face model ID.
- `--max_tokens`: Maximum new tokens to generate (default: 500).

## Example
```
Loading model: HuggingFaceTB/SmolLM2-135M-Instruct...
Chat started! Type 'exit' or 'quit' to end the conversation.
--------------------------------------------------
You: Hello, who are you?
Assistant: I am a helpful and concise assistant. How can I help you today?
```
