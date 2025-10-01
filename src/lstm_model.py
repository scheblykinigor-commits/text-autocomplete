import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMAutocomplete, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding слой
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM слои
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Выходной слой
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear слой
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def generate(self, input_sequence, max_length=20, temperature=1.0):
        """Генерация продолжения текста"""
        self.eval()
        
        with torch.no_grad():
            generated = input_sequence.copy()
            
            # Начальное скрытое состояние
            hidden = None
            
            for _ in range(max_length):
                # Подготовка входных данных
                input_tensor = torch.tensor([generated], dtype=torch.long)
                
                # Forward pass
                output, hidden = self.forward(input_tensor, hidden)
                
                # Получение предсказания для последнего токена
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Выбор следующего токена (жадный выбор)
                next_token = torch.argmax(next_token_probs).item()
                
                # Добавление токена к последовательности
                generated.append(next_token)
                
                # Остановка если достигли конца последовательности (условно)
                if next_token == 0:  # PAD token
                    break
                    
            return generated