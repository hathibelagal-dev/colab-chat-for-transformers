# Project: Colab Chat for Transformers

A minimal, secure-by-default CLI chat application for interacting with instruction-tuned LLMs using the Hugging Face `transformers` library.

## Current Progress
- **Chat Engine**: Modern `text-generation` pipeline with support for any model from the Hub.
- **Session Management**: Automated JSON-based saving and loading of conversation history.
- **Core System Tools**: Powerful, opt-in tools for:
  - `calculate`: Math expression evaluation.
  - `run_shell_command`: Bash command execution.
  - `read_file`: File content reading.
  - `write_file`: File creation and updates.
- **Security Model**: Tools are disabled by default. Individual flags or `--yolo` required for activation.
- **Customization**: Supports custom system prompts from text files via `--system_prompt`.
- **Distribution**: Installable package with `colab_chat` console command.

## Next Steps
- Consider adding streaming support (currently skipped for UI simplicity with tool calls).
- Support for multi-line user input.
- Enhanced error handling for complex shell command outputs.
- Transition to `pyproject.toml` for modern packaging.

## Usage
```bash
# Install
pip install .

# Start Chat
colab_chat --yolo
```
