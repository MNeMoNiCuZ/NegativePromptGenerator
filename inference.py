import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_negative_prompt(model_path, weights_file, input_prompt):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained(model_path, use_auth_token=True)
    
    # Load the weights from weights.pt into the model
    model_weights = torch.load(weights_file)
    model.load_state_dict(model_weights)

    model.eval()  # Set the model to evaluation mode

    # Encode the input prompt to tensor
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

    # Generate a sequence of tokens from the input
    output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)

    # Decode the generated tokens to a string
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py 'Your input prompt here'")
        sys.exit(1)

    input_prompt = sys.argv[1]
    model_path = 'models/negativeprompt_054'
    weights_file = 'models/negativeprompt_054/weights.pt'

    negative_prompt = generate_negative_prompt(model_path, weights_file, input_prompt)
    print(f"Generated Negative Prompt: {negative_prompt}")
