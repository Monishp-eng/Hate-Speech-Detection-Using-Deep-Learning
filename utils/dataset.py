import torch
from torch.utils.data import Dataset, DataLoader

class HateSpeechDataset(Dataset):
    """
    PyTorch Dataset wrapper for our tokenized and padded sequences.
    Inheriting from Dataset allows us to use DataLoader for automatic batching.
    """
    def __init__(self, sequences, labels):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_data_loader(x_data, y_data, batch_size, shuffle=True):
    dataset = HateSpeechDataset(x_data, y_data)
    # DataLoader handles shuffling, batching, and dropping incomplete last batches if desired
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
