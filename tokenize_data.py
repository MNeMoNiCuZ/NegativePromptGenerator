import os
import torch
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence

# Directories
positive_prompts_dir = 'dataset/negative'
negative_prompts_dir = 'dataset/negative'

# Special token for separation
separator_token = " <|endoftext|> "  # Use the GPT-2's end-of-text token

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_and_format(positive_dir, negative_dir, tokenizer, max_length=1024):
    formatted_data = []
    
    # Ensure the tokenizer has a padding token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for filename in os.listdir(positive_dir):
        if filename.endswith('.txt'):
            positive_path = os.path.join(positive_dir, filename)
            negative_path = os.path.join(negative_dir, filename)

            with open(positive_path, 'r', encoding='utf-8') as pos_file, open(negative_path, 'r', encoding='utf-8') as neg_file:
                positive_prompt = pos_file.read().strip()
                negative_prompt = neg_file.read().strip()

                # Use the explicitly defined separator token
                combined_prompt = positive_prompt + separator_token + negative_prompt
                # Truncate and encode the combined prompt
                tokenized_prompt = tokenizer.encode(combined_prompt, add_special_tokens=True, max_length=max_length, truncation=True, return_tensors='pt')
                
                formatted_data.append(tokenized_prompt.squeeze(0))  # Remove the batch dimension

    # Pad the sequences so they all have the same length
    padded_data = pad_sequence(formatted_data, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return padded_data

# Main part of your script
tokenized_data = tokenize_and_format(positive_prompts_dir, negative_prompts_dir, tokenizer)
torch.save(tokenized_data, 'dataset/tokenized_data.pt')
print(f"Tokenized and formatted data saved as tokenized_data.pt")