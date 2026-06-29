import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
UTILS_DIR = os.path.join(BASE_DIR, 'utils')

# Create necessary directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
MAX_LEN = 100            # Maximum sequence length
MAX_WORDS = 20000        # Vocabulary size limit
EMBEDDING_DIM = 200      # Size of word embeddings
HIDDEN_DIM = 128         # Number of features in LSTM hidden state
NUM_CLASSES = 3          # Hate Speech = 0, Offensive Language = 1, Neither = 2
DROPOUT_RATE = 0.4       # Dropout probability for regularization
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.001

# Files
TOKENIZER_PATH = os.path.join(PROCESSED_DATA_DIR, 'tokenizer.pkl')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_bilstm_attention.pth')
