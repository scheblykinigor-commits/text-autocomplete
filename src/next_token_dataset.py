import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
import numpy as np

class NextTokenDataset(Dataset):
    def __init__(self, data_path, seq_length=20):
        """
        Dataset for next token prediction
        
        Args:
            data_path: Path to CSV file with tokens
            seq_length: Length of input sequences
        """
        self.df = pd.read_csv(data_path)
        self.seq_length = seq_length
        
        self.df['tokens'] = self.df['tokens'].apply(eval)
        
        self._build_vocab()
        
        self.sequences = self._prepare_sequences()
    
    def _build_vocab(self):
        """Создаем словарный запас из всех токенов"""
        all_tokens = []
        for tokens in self.df['tokens']:
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        self.vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_counts.most_common(10000))}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        self.vocab_size = len(self.vocab)
    
    def _prepare_sequences(self):
        """Подготовка последовательности тренировок"""
        sequences = []
        for tokens in self.df['tokens']:
            if len(tokens) < 2:
                continue
                
            token_indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            
            for i in range(len(token_indices) - self.seq_length):
                input_seq = token_indices[i:i + self.seq_length]
                target_seq = token_indices[i + 1:i + self.seq_length + 1]
                sequences.append((input_seq, target_seq))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )
    
    def decode_tokens(self, indices):
        """Convert indices back to tokens"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        tokens = []
        for idx in indices:
            if idx == self.vocab['<pad>']:
                break
            tokens.append(self.idx_to_token.get(idx, '<unk>'))
        
        return tokens