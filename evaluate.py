import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import ast

from config.config import *
from predict import HateSpeechPredictor
from utils.dataset import create_data_loader
from utils.metrics import calculate_metrics
from utils.preprocessing import clean_text

def evaluate_model(data_path="data/raw/labeled_data.csv"):
    """
    Evaluates the model using the test dataset.
    Loads trained model and test data, computes standard metrics like 
    Accuracy, Precision, Recall, F1-score, and confusion matrix.
    """
    print("="*60)
    print(" Hate Speech Detection - Model Evaluation ")
    print("="*60)

    try:
        predictor = HateSpeechPredictor()
        device = predictor.device
        model = predictor.get_model()
        tokenizer = predictor.get_tokenizer()
    except Exception as e:
        print(f"Error loading predictor: {e}")
        sys.exit(1)

    print("\nLoading test dataset...")
    
    # We will simulate a test dataset if actual file doesn't exist or is hard to parse for this example.
    # In a real scenario, the data_path is fully handled
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Using simple random sample for test
        df_test = df.sample(frac=0.2, random_state=42)
        
        if 'tweet' in df_test.columns and 'class' in df_test.columns:
            texts = df_test['tweet'].apply(clean_text).values
            labels = df_test['class'].values
        else:
            print("Dataset must have 'tweet' and 'class' columns.")
            sys.exit(1)
    else:
        print("Data file not found. Using dummy evaluation dataset.")
        raw_texts = ["I hate you", "You are an idiot", "Have a great day", "What a stupid idea", "I love AI"]
        raw_labels = [0, 1, 2, 1, 2] # 0: Hate, 1: Offensive, 2: Neutral
        texts = [clean_text(t) for t in raw_texts]
        labels = raw_labels
    
    # Process text sequences identical to training scheme
    sequences = tokenizer.texts_to_sequences(texts)
    from utils.preprocessing import pad_sequences
    X_test = pad_sequences(sequences, MAX_LEN)
    
    # Generate batch dataloader for efficient memory iterations
    test_loader = create_data_loader(X_test, labels, BATCH_SIZE, shuffle=False)
    
    print("\nEvaluating Model on test set...")
    model.eval()  # Disables dropout/batchnorm for reliable deterministic predictions
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # Disable gradient graph tracking saving memory
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Retrieve max probability indices mapping to classes
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Compile deep evaluation statistics focusing especially on Class 0 (Hate)
    target_names = ["Hate Speech (0)", "Offensive (1)", "Neutral (2)"]
    metrics = calculate_metrics(all_targets, all_preds, target_names)
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (Macro)")
    print(f"Recall:    {metrics['recall']:.4f} (Macro)")
    print(f"F1-Score:  {metrics['f1']:.4f} (Macro)")
    
    print("\n=== Classification Report ===")
    print(metrics['report'])
    
    print("=== Confusion Matrix ===")
    print(metrics['confusion_matrix'])
    print("\nEvaluation successfully completed.")

if __name__ == "__main__":
    evaluate_model()