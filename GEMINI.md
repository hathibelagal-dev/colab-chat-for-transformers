# Gemini Project Context: Colab Chat for Transformers

This project is a minimal, secure-by-default CLI chat application for interacting with instruction-tuned LLMs using the Hugging Face `transformers` library.

## Project Overview

- **Purpose**: Provides an interactive terminal interface to chat with any model from the Hugging Face Hub that supports chat templates.
- **Key Features**:
    - Interactive CLI chat with streaming support.
    - **GGUF Support**: Native support for GGUF files via automatic detection and the `--gguf_file` flag.
    - Opt-in system tools (calculation, shell execution, file read/write).
    - Automatic session saving and loading (JSON format).
    - Device mapping (CPU/GPU) via `accelerate`.
- **Primary Technologies**:
    - **Language**: Python (>= 3.8)
    - **Libraries**: `transformers`, `torch`, `accelerate`, `gguf`
    - **Packaging**: `setuptools` (with planned transition to `pyproject.toml`).

## Architecture

The project is centered around a single source file:
- `chat.py`: Contains the main application logic, including:
    - `Core Tools`: Functions for `calculate`, `run_shell_command`, `read_file`, and `write_file`.
    - `Session Management`: Logic to save/load conversation history to/from `sessions/`.
    - `Chat Loop`: An iterative loop that manages user input, model generation (using `transformers` pipeline), and tool execution.

## Building and Running

### Development Setup
```bash
# Clone and install in editable mode
pip install -e .
```

### Running the Application
```bash
# Basic chat
colab_chat

# Chat with a local GGUF model (auto-detects .gguf extension)
colab_chat --model path/to/model.gguf

# Chat with a GGUF model from the Hub
colab_chat --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf_file model.gguf

# Chat with all tools enabled (YOLO mode)
colab_chat --yolo
```

### Testing
- **TODO**: No automated tests currently exist. Future development should include unit tests for core tools and integration tests for the chat loop.

## Development Conventions

- **Security First**: System tools (shell, read, write) are disabled by default. Do not change this default behavior.
- **Tool Implementation**: Tools are implemented as simple Python functions and mapped dynamically based on CLI flags.
- **Session Format**: Sessions are stored as JSON files containing the model ID, timestamp, and the full message history (including tool calls and results).
- **Styling**: Adhere to standard Python PEP 8 guidelines. Use type hints for new functions.
- **Documentation**: Keep `README.md` and `PROJECT.md` updated with new features and progress.

## Key Files

- `chat.py`: Main entry point and application logic.
- `setup.py`: Packaging and dependency configuration.
- `requirements.txt`: List of required Python packages.
- `PROJECT.md`: High-level project status and roadmap.
- `README.md`: User-facing documentation.
- `sessions/`: (Generated) Directory for stored chat histories.
