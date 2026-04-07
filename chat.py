import argparse
import torch
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="A minimal CLI chat application using transformers.")
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
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    
    # Initialize the text-generation pipeline
    # device_map="auto" will use GPU if available
    try:
        pipe = pipeline(
            "text-generation", 
            model=args.model, 
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
    ]

    print("\nChat started! Type 'exit' or 'quit' to end the conversation.")
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
            
            # The assistant's response is the last message in the returned list
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            
            print(f"Assistant: {assistant_response}")
            
            # Update history
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
