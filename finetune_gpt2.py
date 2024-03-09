from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
import os
import json

# Training settings
EPOCHS = 100
BATCH_SIZE = 2
WORKERS = 8
LEARNING_RATE = 5e-5
PROJECT_NAME = "negativeprompt"
GRADIENT_ACCUMULATION_STEPS = 4  # Adjust based on your GPU memory

# Paths
TOKENIZED_DATA_PATH = 'dataset/tokenized_data.pt'
MODEL_SAVE_PATH = 'models/'
# Resume training from CHECKPOINT_PATH
CHECKPOINT_PATH = 'models/negativeprompt_054/'  # e.g., 'models/your_checkpoint_folder/'

def load_dataset(file_path):
    return torch.load(file_path)

def create_attention_masks(tokenized_texts, tokenizer):
    return tokenized_texts.ne(tokenizer.pad_token_id).int()

def train(model, tokenizer, dataset, attention_masks, start_epoch, epochs, batch_size, learning_rate, workers, device):
    # Setup DataLoader for batched training
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=workers)

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)

    # Set model to training mode
    model.train()  

    # Path for logging training progress
    log_file_path = os.path.join(MODEL_SAVE_PATH, f"{PROJECT_NAME}_log.txt")
    with open(log_file_path, 'a') as log_file:
        for epoch in range(start_epoch, epochs):
            # Reset gradients
            optimizer.zero_grad()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, (batch, _) in enumerate(progress_bar):
                # Move batch to the device
                batch = batch.to(device)
                # Prepare attention masks
                mask = attention_masks[:batch.size(0)].to(device)

                # Forward pass
                outputs = model(batch, attention_mask=mask, labels=batch)
                # Scale loss
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                # Backward pass
                loss.backward()
                total_loss += loss.item()

                # Step the optimizer and scheduler
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                    optimizer.step()
                    scheduler.step()
                    # Clear gradients
                    optimizer.zero_grad()
                    progress_bar.set_postfix({'loss': loss.item() * GRADIENT_ACCUMULATION_STEPS})

            # Log average loss for the epoch
            avg_loss = total_loss / (len(dataloader) // GRADIENT_ACCUMULATION_STEPS)
            log_file.write(f"Epoch: {epoch + 1}, Avg Loss: {avg_loss}\n")
            print(f"Epoch: {epoch + 1}, Avg Loss: {avg_loss}")

            # Save model and tokenizer at the end of each epoch
            checkpoint_dir = f"{MODEL_SAVE_PATH}{PROJECT_NAME}_{str(epoch + 1).zfill(len(str(epochs)))}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save training progress
            training_progress = {'epoch': epoch}
            with open(os.path.join(checkpoint_dir, 'training_progress.json'), 'w') as f:
                json.dump(training_progress, f)
            print(f"Checkpoint saved to {checkpoint_dir}")

            # Save model weights
            weights_path = f"{checkpoint_dir}/weights.pt"
            torch.save(model.state_dict(), weights_path)
            print(f"Model weights saved to {weights_path}")

def main():
    try:
        # Setup device for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Check if a checkpoint exists for resuming training
        if CHECKPOINT_PATH and os.path.isdir(CHECKPOINT_PATH):
            print(f"Loading model and tokenizer from checkpoint: {CHECKPOINT_PATH}")
            model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_PATH)
            # Load the saved training progress
            with open(os.path.join(CHECKPOINT_PATH, 'training_progress.json'), 'r') as f:
                training_progress = json.load(f)
            start_epoch = training_progress['epoch']
        else:
            # Initialize model and tokenizer from pretrained configuration if no checkpoint found
            print("Initializing model and tokenizer from 'gpt2' pretrained model.")
            model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            start_epoch = 0

        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset and prepare DataLoader
        tokenized_texts = load_dataset(TOKENIZED_DATA_PATH)
        attention_masks = create_attention_masks(tokenized_texts, tokenizer)
        dataset = TensorDataset(tokenized_texts, torch.zeros_like(tokenized_texts))

        # Train the model
        train(model, tokenizer, dataset, attention_masks, start_epoch, EPOCHS, BATCH_SIZE, LEARNING_RATE, WORKERS, device)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
