import re
import string
import pickle
from collections import Counter
import torch
import numpy as np

def clean_text(text):
    """
    Cleans raw text data by removing URLs, HTML tags, punctuation, and numbers.
    Converts to lowercase.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text)               # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text)                   # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)             # Remove words containing numbers
    return text.strip()

class TextTokenizer:
    """
    A simple custom Tokenizer to build vocabulary, convert texts to sequences,
    and pad them for uniform Neural Network input.
    """
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2

    def fit_on_texts(self, texts):
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        # Sort by frequency
        common_words = word_counts.most_common(self.num_words - 2) if self.num_words else word_counts.most_common()
        
        # Build vocabulary mapping word -> integer index
        for word, _ in common_words:
            if word not in self.word_index:
                self.word_index[word] = self.vocab_size
                self.vocab_size += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(w, self.word_index["<UNK>"]) for w in text.split()]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, max_len):
    """
    Pads sequences with 0 (<PAD>) so all inputs to the network match `max_len`.
    If longer, truncate. If shorter, append 0s.
    """
    features = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            features[i, :min(len(seq), max_len)] = np.array(seq)[:max_len]
    return features

def save_tokenizer(tokenizer, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(tokenizer, handle)

def load_tokenizer(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)
