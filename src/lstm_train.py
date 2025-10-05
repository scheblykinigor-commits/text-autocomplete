import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from lstm_model import LSTMAutoComplete
from next_token_dataset import NextTokenDataset
from eval_lstm import calculate_rouge_lstm

def train_model(model, train_loader, val_loader, vocab, device, num_epochs=10, lr=0.001):
    """
    Обучаем модель LSTM
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_rouge_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(inputs)
            
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        rouge1, rouge2, examples = calculate_rouge_lstm(model, val_loader, vocab, device)
        val_rouge_scores.append(rouge1)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val ROUGE-1: {rouge1:.4f}')
        print(f'  Val ROUGE-2: {rouge2:.4f}')
        print('  Example predictions:')
        for i, (input_text, pred_text, target_text) in enumerate(examples[:2]):
            print(f'    Input: {" ".join(input_text)}')
            print(f'    Pred: {" ".join(pred_text)}')
            print(f'    Target: {" ".join(target_text)}')
            print()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_rouge_scores)
    plt.title('Validation ROUGE-1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE-1')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, train_losses, val_rouge_scores

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = NextTokenDataset('F:/text-autocomplete/data/train.csv')
    val_dataset = NextTokenDataset('F:/text-autocomplete/data/val.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = LSTMAutoComplete(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model, train_losses, val_rouge_scores = train_model(
        model, train_loader, val_loader, train_dataset.vocab, device,
        num_epochs=10, lr=0.001
    )
    
    os.makedirs('F:/text-autocomplete/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': train_dataset.vocab,
        'config': {
            'vocab_size': train_dataset.vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 128,
            'num_layers': 2
        }
    }, 'F:/text-autocomplete/models/lstm_model.pth')
    
    print("Модель сохранена")

if __name__ == "__main__":
    main()