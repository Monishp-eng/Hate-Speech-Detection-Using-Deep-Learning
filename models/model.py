import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism for Classification.
    This architecture learns sequential context from both directions and
    weighs the relative importance of words (attention) before classifying.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Embedding Layer: Learns dense vector representations mapping words to meaningful continuous spaces
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # 2. Bidirectional LSTM: Captures temporal dependencies by running LSTM forward & backward
        # batch_first=True expects inputs as [batch_size, seq_len, features]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            num_layers=2, bidirectional=True, 
                            batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        # 3. Attention mechanism weights: Learns what parts of the sequence are vital for classification
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        
        # 4. Fully Connected Layers & Dropout
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layer maps to our class logits (scores for each class before softmax)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embed_dropout(self.embedding(text))
        
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2] (since bidirectional)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Attention scoring and softmax to get probabilities across the sequence
        # attn_scores shape: [batch_size, seq_len, 1]
        attn_scores = self.attention_weights(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1) 
        
        # Multiply LSTM hidden states by their calculated attention weights
        # Context vector shape: [batch_size, hidden_dim * 2]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Pass context vector through fully connected layers to get final logits
        dense_out = F.relu(self.fc(self.fc_dropout(context_vector)))
        logits = self.out(dense_out)
        
        return logits
