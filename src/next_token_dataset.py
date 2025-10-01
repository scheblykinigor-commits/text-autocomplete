import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
import numpy as np

class NextTokenDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_length=50):
        self.df = pd.read_csv(data_path)
        self.df['tokens'] = self.df['tokens'].apply(eval)
        self.max_length = max_length
        
        # Создание или использование существующего словаря
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
            
        self.vocab_size = len(self.vocab)
        
    def _build_vocab(self):
        """Построение словаря"""
        all_tokens = []
        for tokens in self.df['tokens']:
            all_tokens.extend(tokens)
            
        token_counts = Counter(all_tokens)
        vocab = {token: idx + 2 for idx, (token, count) in enumerate(token_counts.items())}  # +2 для резервирования индексов
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        
        return vocab
    
    def text_to_indices(self, tokens):
        """Преобразование токенов в индексы"""
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]['tokens']
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Создание входной последовательности и цели (сдвиг на 1 токен)
        input_indices = self.text_to_indices(tokens[:-1])
        target_indices = self.text_to_indices(tokens[1:])
        
        # Добавление паддинга
        while len(input_indices) < self.max_length - 1:
            input_indices.append(self.vocab['<PAD>'])
            target_indices.append(self.vocab['<PAD>'])
        
        return {
            'input_ids': torch.tensor(input_indices, dtype=torch.long),
            'target_ids': torch.tensor(target_indices, dtype=torch.long),
            'length': len(tokens) - 1
        }