import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Ensure module path imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from models.model import BiLSTMWithAttention
from utils.dataset import create_data_loader
from utils.preprocessing import clean_text, TextTokenizer, pad_sequences, save_tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split # Note: We will use dummy data logic to illustrate

def train_model(data_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Load & Preprocess Data
    # For demonstration, we simulate dataset loading if None is passed. 
    # Replace with pandas read csv (e.g., pd.read_csv('hate_speech.csv'))
    print("Loading data...")
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        texts = df['tweet'].apply(clean_text).values
        labels = df['class'].values # e.g., 0: Hate, 1: Offensive, 2: Neutral
    else:
        print("Using dummy dataset for illustration...")
        raw_texts = ["I hate you", "You are offensive", "Have a great day", "Stupid idiot", "I love AI"]
        raw_labels = [0, 1, 2, 0, 2]
        texts = [clean_text(t) for t in raw_texts]
        labels = raw_labels

    # Tokenization
    print("Fitting tokenizer...")
    tokenizer = TextTokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    
    # Sequences & Padding
    sequences = tokenizer.texts_to_sequences(texts)
    X_data = pad_sequences(sequences, MAX_LEN)
    
    # Train-test split (simplified: 80/20)
    split_idx = int(0.8 * len(X_data))
    X_train, y_train = X_data[:split_idx], labels[:split_idx]
    X_val, y_val = X_data[split_idx:], labels[split_idx:]
    
    train_loader = create_data_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    
    # Model Initialization
    # Pass tokenizer.vocab_size ensuring we have all found words + padding + unk tokens
    model = BiLSTMWithAttention(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE).to(device)
    
    # Loss Function (Cross Entropy calculates probabilities internally and checks against true class indices)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Adam applies adaptive learning rates per parameter, robust for NLP problems)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Training iteration processing batches
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset gradients before backward pass
            optimizer.zero_grad()
            
            # Forward pass: obtain raw class logits
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Compute gradients with backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation phase: Turn off dropout and evaluate model performance
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss/len(train_loader) if len(train_loader) else 0
        avg_val_loss = val_loss/len(val_loader) if len(val_loader) else 0
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Checkpointing: Save parameters capturing best validation performance
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)

if __name__ == "__main__":
    train_model(data_path="data/raw/labeled_data.csv")
