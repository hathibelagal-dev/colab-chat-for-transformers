import argparse
import json
import os
import torch
import subprocess
from datetime import datetime
from transformers import pipeline, TextStreamer

# --- Core Tools ---

def calculate(expression: str):
    """
    Evaluate a simple mathematical expression.

    Args:
        expression: The math expression to evaluate, e.g. '2 + 2', '10 * 5'.
    """
    try:
        # Use simple eval for demonstration; in production, use a safer method
        result = eval(expression, {"__builtins__": None}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

def run_shell_command(command: str):
    """
    Execute a bash shell command and return its output.

    Args:
        command: The shell command to execute, e.g., 'ls -l', 'grep pattern file.txt'.
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds."}
    except Exception as e:
        return {"error": str(e)}

def read_file(path: str):
    """
    Read the contents of a file.

    Args:
        path: The absolute or relative path to the file.
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
        return {"content": content, "path": path}
    except Exception as e:
        return {"error": str(e), "path": path}

def write_file(path: str, content: str):
    """
    Write content to a file, overwriting it if it already exists.

    Args:
        path: The absolute or relative path where the file should be written.
        content: The text content to write to the file.
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, 'w') as f:
            f.write(content)
        return {"status": "success", "path": path}
    except Exception as e:
        return {"error": str(e), "path": path}

# --- Helper Functions ---

def save_session(messages, filename, model_name):
    """Saves the conversation history to a JSON file."""
    data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_session(filename):
    """Loads a conversation history from a JSON file."""
    if not os.path.exists(filename):
        print(f"Error: Session file '{filename}' not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="A minimal CLI chat application using transformers with opt-in core tool-use support.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="The model ID (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (default: 8192)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to a JSON session file to load"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable automatic saving of the chat session"
    )
    parser.add_argument(
        "--session_name",
        type=str,
        help="Custom name for the session file"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="Path to a text file containing a custom system prompt"
    )
    
    # Tool-enabling flags
    parser.add_argument("--calculate", action="store_true", help="Enable the calculation tool")
    parser.add_argument("--shell", action="store_true", help="Enable the shell command tool")
    parser.add_argument("--read", action="store_true", help="Enable the file reading tool")
    parser.add_argument("--write", action="store_true", help="Enable the file writing tool")
    parser.add_argument("--yolo", action="store_true", help="Enable all tools at once (BE CAREFUL)")
    
    args = parser.parse_args()

    # Load core tools based on opt-in flags
    tools = []
    if args.calculate or args.yolo: tools.append(calculate)
    if args.shell or args.yolo: tools.append(run_shell_command)
    if args.read or args.yolo: tools.append(read_file)
    if args.write or args.yolo: tools.append(write_file)
    
    tool_map = {t.__name__: t for t in tools}

    # Setup session file
    session_dir = "sessions"
    if not args.no_save and not os.path.exists(session_dir):
        os.makedirs(session_dir)

    system_content = "You are a helpful assistant. You have access to tools to help answer questions if needed."
    if args.system_prompt:
        try:
            with open(args.system_prompt, 'r') as f:
                system_content = f.read().strip()
            print(f"Loaded system prompt from: {args.system_prompt}")
        except Exception as e:
            print(f"Error loading system prompt file: {e}. Using default.")

    messages = [
        {"role": "system", "content": system_content},
    ]
    
    current_model = args.model
    session_file = None

    if args.load:
        session_data = load_session(args.load)
        if session_data:
            messages = session_data["messages"]
            loaded_model = session_data.get("model", "unknown")
            print(f"Restored session from: {args.load}")
            if loaded_model != current_model:
                print(f"Warning: Loaded session used model '{loaded_model}', but you are using '{current_model}'.")
            session_file = args.load
            
            print("-" * 50)
            for msg in messages:
                if msg["role"] == "system": continue
                if "content" in msg and msg["content"]:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    print(f"{role}: {msg['content']}")
                elif "tool_calls" in msg:
                    print(f"Assistant (Tool Calls): {[tc['function']['name'] for tc in msg['tool_calls']]}")
                elif msg["role"] == "tool":
                    print(f"Tool Result ({msg['name']}): {msg['content']}")
            print("-" * 50)

    if not session_file and not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.session_name if args.session_name else f"chat_{timestamp}"
        session_file = os.path.join(session_dir, f"{name}.json")

    print(f"Loading model: {current_model}...")
    
    try:
        pipe = pipeline(
            "text-generation", 
            model=current_model, 
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        # Configure generation defaults to avoid warnings
        # We set these on the model's generation_config once
        pipe.model.generation_config.max_new_tokens = args.max_tokens
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
        # Silence the 'max_length' vs 'max_new_tokens' warning
        pipe.model.generation_config.max_length = None
        
        # Initialize streamer only if no tools are enabled
        streamer = None
        if not tools:
            streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nChat started! Type 'exit' or 'quit' to end the conversation.")
    if tools:
        tool_names = [t.__name__ for t in tools]
        print(f"Tools enabled: {tool_names}")
        # Only show the warning if shell, read, or write tools are enabled
        if any([args.shell, args.read, args.write, args.yolo]):
            print("\033[91mWARNING: This model has access to your shell and/or file system. Proceed with caution.\033[0m")
    else:
        print("No tools enabled.")

    if session_file and not args.no_save:
        print(f"This session will be saved to: {session_file}")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # Tool-use loop
            while True:
                # If streaming is enabled, print the prefix first
                if streamer and not (messages[-1]["role"] == "tool"):
                    print("Assistant: ", end="", flush=True)

                # Call pipeline without generation-related args to avoid warnings
                # as they are now set in pipe.model.generation_config
                outputs = pipe(
                    messages, 
                    tools=tools if tools else None,
                    streamer=streamer
                )
                
                assistant_message = outputs[0]["generated_text"][-1]
                
                if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                    # Tool calls aren't streamed by our logic, so we show them as before
                    messages.append(assistant_message)
                    
                    for tool_call in assistant_message["tool_calls"]:
                        fn_name = tool_call["function"]["name"]
                        fn_args = tool_call["function"]["arguments"]
                        
                        print(f"[*] Calling tool: {fn_name}({fn_args})")
                        
                        if fn_name in tool_map:
                            try:
                                result = tool_map[fn_name](**fn_args)
                            except Exception as e:
                                result = {"error": str(e)}
                        else:
                            result = {"error": f"Tool '{fn_name}' not found."}
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": fn_name,
                            "content": json.dumps(result)
                        })
                    
                    continue
                else:
                    # If not streaming, print the full response now
                    if not streamer:
                        assistant_response = assistant_message.get("content", "")
                        print(f"Assistant: {assistant_response}")
                    else:
                        # Streamer finished, add a newline
                        print()
                    
                    messages.append(assistant_message)
                    break

            if session_file and not args.no_save:
                save_session(messages, session_file, current_model)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
