import argparse
import json
import os
import torch
import importlib.util
from datetime import datetime
from transformers import pipeline

# --- Default Tools ---

def get_weather(location: str):
    """
    Get the current weather in a given location.

    Args:
        location: The city name, e.g. London, Tokyo, San Francisco.
    """
    # Mock weather data
    return {"location": location, "temperature": "22°C", "condition": "Sunny"}

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

DEFAULT_TOOLS = [get_weather, calculate]

# --- Helper Functions ---

def load_tools_from_file(file_path):
    """Loads a list of tool functions from a Python file."""
    if not os.path.exists(file_path):
        print(f"Error: Tools file '{file_path}' not found.")
        return []
    
    spec = importlib.util.spec_from_file_location("external_tools", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Assume the user defines a 'tools' list in their file
    if hasattr(module, "tools") and isinstance(module.tools, list):
        return module.tools
    else:
        # Or just pick up all callable functions that have a docstring
        return [getattr(module, name) for name in dir(module) 
                if callable(getattr(module, name)) and not name.startswith("_") 
                and getattr(module, name).__doc__]

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
    parser = argparse.ArgumentParser(description="A minimal CLI chat application using transformers with tool-use support.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="The model ID (default: HuggingFaceTB/SmolLM2-1.7B-Instruct, which supports tools)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate (default: 500)"
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
        "--no_tools",
        action="store_true",
        help="Disable tool-use functionality"
    )
    parser.add_argument(
        "--tools_file",
        type=str,
        help="Path to a Python file containing custom tool functions"
    )
    args = parser.parse_args()

    # Load tools
    tools = []
    if not args.no_tools:
        if args.tools_file:
            tools = load_tools_from_file(args.tools_file)
        else:
            tools = DEFAULT_TOOLS
    
    tool_map = {t.__name__: t for t in tools}

    # Setup session file
    session_dir = "sessions"
    if not args.no_save and not os.path.exists(session_dir):
        os.makedirs(session_dir)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. You have access to tools to help answer questions if needed."},
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
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nChat started! Type 'exit' or 'quit' to end the conversation.")
    if tools:
        print(f"Tools enabled: {[t.__name__ for t in tools]}")
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

            # Tool-use loop: The model might call tools multiple times
            while True:
                # Call pipeline
                # 'tools' arg is only passed if the model supports it and we have tools
                outputs = pipe(
                    messages, 
                    tools=tools if tools else None,
                    max_new_tokens=args.max_tokens,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
                
                assistant_message = outputs[0]["generated_text"][-1]
                
                if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                    # Model wants to call tools
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
                    
                    # After adding tool results, loop back to let the model generate the final response
                    continue
                else:
                    # Model provided a text response
                    assistant_response = assistant_message.get("content", "")
                    print(f"Assistant: {assistant_response}")
                    messages.append(assistant_message)
                    break

            # Save session after each turn
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
