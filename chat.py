import argparse
import json
import os
import torch
from datetime import datetime
from transformers import pipeline

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
    parser = argparse.ArgumentParser(description="A minimal CLI chat application using transformers with save/load support.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="The model ID from Hugging Face Hub (default: HuggingFaceTB/SmolLM2-135M-Instruct)"
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
        help="Custom name for the session file (e.g., 'my_chat')"
    )
    args = parser.parse_args()

    # Setup session file
    session_dir = "sessions"
    if not args.no_save and not os.path.exists(session_dir):
        os.makedirs(session_dir)

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
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
            
            # Use the loaded file as the current session file unless a new name is specified
            session_file = args.load
            
            # Show history
            print("-" * 50)
            for msg in messages:
                if msg["role"] != "system":
                    role = "You" if msg["role"] == "user" else "Assistant"
                    print(f"{role}: {msg['content']}")
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

            # Generate response
            outputs = pipe(
                messages, 
                max_new_tokens=args.max_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            print(f"Assistant: {assistant_response}")
            
            messages.append({"role": "assistant", "content": assistant_response})

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
