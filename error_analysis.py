import os
import sys
import torch
import pandas as pd

from config.config import *
from predict import HateSpeechPredictor
from utils.preprocessing import clean_text

def analyze_errors(data_path="data/raw/labeled_data.csv", num_samples=100):
    """
    Error Analysis Module:
    Shows misclassified examples, compares predicted vs actual labels,
    and identifies patterns like slang or sarcasm.
    """
    print("="*60)
    print("        Hate Speech - Error Analysis         ")
    print("="*60)

    predictor = HateSpeechPredictor()
    class_mapping = predictor.class_mapping

    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        # Dummy data for demonstration
        texts = ["I love you so much", "What a stupid post", "Have a great day", "I hate everything you do"]
        labels = [0, 1, 2, 2] # Some intentionally wrong labels to simulate errors
    else:
        df = pd.read_csv(data_path)
        df_sample = df.sample(min(num_samples, len(df)), random_state=42)
        if 'tweet' in df_sample.columns and 'class' in df_sample.columns:
            texts = df_sample['tweet'].tolist()
            labels = df_sample['class'].tolist()
        else:
            print("Dataset must have 'tweet' and 'class' columns.")
            return

    print(f"\nRunning predictions on {len(texts)} samples...")
    results = predictor.predict_batch(texts)

    misclassified = []
    
    for i, res in enumerate(results):
        actual_idx = labels[i]
        actual_label = class_mapping.get(actual_idx, "Unknown")
        predicted_label = res['prediction']
        
        if actual_label != predicted_label:
            misclassified.append({
                'text': texts[i],
                'actual': actual_label,
                'predicted': predicted_label,
                'confidence': res['confidence']
            })

    print(f"\nFound {len(misclassified)} errors out of {len(texts)} samples analyzed.\n")
    
    if not misclassified:
        print("Model performed perfectly on this sample!")
        return

    # Print top 15 errors
    print("=== Misclassified Examples ===")
    for i, err in enumerate(misclassified[:15]):
        print(f"\n[Example {i+1}]")
        print(f"Original Text: '{err['text']}'")
        print(f"Actual Label:  {err['actual']}")
        print(f"Predicted:     {err['predicted']} (Conf: {err['confidence']:.2f})")
    
    print("\n--- Insights & Patterns ---")
    print("Look closely at the misclassified examples to observe common patterns:")
    print("1. Sarcasm / Irony: Often predicted incorrectly due to contradictory phrasing.")
    print("2. Implicit vs Explicit Slang: Context-heavy internet slang may mislead the model without broad training context.")
    print("3. Retweets (RT) & Mentions: The noise from metadata limits context clarity.")

if __name__ == "__main__":
    analyze_errors()