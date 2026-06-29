import os
import sys
import torch
import torch.nn.functional as F
import logging
import argparse
import json

from config.config import *
from models.model import BiLSTMWithAttention
from utils.preprocessing import clean_text, load_tokenizer, pad_sequences

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HateSpeechPredictor:
    """
    Reusable prediction pipeline for Hate Speech Detection.
    Handles loading the model, preprocessing, and inference.
    """
    def __init__(self, model_path=BEST_MODEL_PATH, tokenizer_path=TOKENIZER_PATH, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.class_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        
        logger.info(f"Initializing Predictor on device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = load_tokenizer(tokenizer_path)
            logger.info("Tokenizer loaded successfully.")
            
            # Initialize model architecture
            self.model = BiLSTMWithAttention(self.tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Vital: Set model to evaluation mode (disables dropout, affects layernorm)
            self.model.eval()
            logger.info("Model weights loaded and set to eval mode.")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise RuntimeError("Ensure the model has been trained and saved correctly.") from e

    def predict_single(self, text: str):
        """
        Predict the class for a single text input.
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list):
        """
        Predict the class for a batch of texts.
        """
        # 1. Preprocess: Clean, tokenization, padding
        cleaned_texts = [clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(seqs, MAX_LEN)
        
        # 2. Convert to Tensor
        input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
        
        results = []
        
        # 3. Inference without gradient tracking (memory efficient)
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # 4. Softmax to get probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
        for i in range(len(texts)):
            class_probs = probs[i]
            max_prob = float(max(class_probs))
            pred_idx = int(class_probs.argmax())
            
            # Threshold Tuning: If max probability is below threshold, we can flag as Uncertain
            if max_prob < self.threshold:
                prediction_label = "Uncertain / Unconfident"
            else:
                prediction_label = self.class_mapping[pred_idx]
                
            results.append({
                "text": texts[i],
                "prediction": prediction_label,
                "confidence": max_prob,
                "probabilities": {self.class_mapping[k]: float(class_probs[k]) for k in self.class_mapping}
            })
            
        return results

    def get_model(self):
        return self.model
        
    def get_tokenizer(self):
        return self.tokenizer

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection CLI")
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for prediction")
    args = parser.parse_args()

    print("="*60)
    print(" Hate Speech Detection - Interactive CLI ")
    print("="*60)
    
    try:
        predictor = HateSpeechPredictor(threshold=args.threshold)
    except Exception:
        sys.exit(1)
        
    print("\nType 'quit' or 'exit' to stop.")
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit']:
                logger.info("Exiting interactive mode.")
                break
                
            if not text:
                continue
                
            # Perform Inference
            result = predictor.predict_single(text)
            
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Class Probabilities: {json.dumps(result['probabilities'], indent=2)}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
