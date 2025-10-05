import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutoComplete(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMAutoComplete, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass for training
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Previous hidden state
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        # Embedding
        x_embed = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(x_embed, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        logits = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def generate(self, input_seq, max_length=20, temperature=1.0):
        """
        Generate text continuation
        
        Args:
            input_seq: Starting sequence tensor of shape (1, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            
        Returns:
            generated_sequence: Generated sequence indices
        """
        self.eval()
        generated = input_seq.clone()
        
        with torch.no_grad():
            hidden = None
            
            for _ in range(max_length):
                logits, hidden = self.forward(generated, hidden)
                next_token_logits = logits[:, -1, :] / temperature
                
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 0:
                    break
        
        return generated
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)