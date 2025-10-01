import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .lstm_model import LSTMAutocomplete
from .next_token_dataset import NextTokenDataset
from .eval_lstm import calculate_rouge

def train_model(train_data_path, val_data_path, vocab_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Функция обучения модели"""
    
    # Параметры
    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    dropout = 0.2
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 256
    
    # Датасеты и даталоадеры
    train_dataset = NextTokenDataset(train_data_path)
    val_dataset = NextTokenDataset(val_data_path, vocab=train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Модель
    model = LSTMAutocomplete(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем PAD token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Обучение
    train_losses = []
    val_rouge_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(input_ids)
            
            # Reshape для вычисления потерь
            output = output.reshape(-1, output.shape[-1])
            target_ids = target_ids.reshape(-1)
            
            # Вычисление потерь
            loss = criterion(output, target_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Средняя потеря за эпоху
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Валидация
        rouge1, rouge2 = calculate_rouge(model, val_loader, device, train_dataset.vocab)
        val_rouge_scores.append((rouge1, rouge2))
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}")
        print("-" * 50)
    
    return model, train_dataset.vocab, train_losses, val_rouge_scores