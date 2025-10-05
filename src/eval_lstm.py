import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def calculate_rouge_lstm(model, data_loader, vocab, device, generation_length=10):
    """
    Рассчитываем баллы ROUGE для модели LSTM
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    idx_to_token = {idx: token for token, idx in vocab.items()}
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating LSTM"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            for i in range(min(32, inputs.size(0))):
                input_seq = inputs[i:i+1]
                target_seq = targets[i]
                
                generated = model.generate(input_seq, max_length=generation_length)
                generated_tokens = generated[0].cpu().numpy()
                
                input_tokens = [idx_to_token.get(idx, '<unk>') for idx in input_seq[0].cpu().numpy() if idx != 0]
                pred_tokens = [idx_to_token.get(idx, '<unk>') for idx in generated_tokens[len(input_seq[0]):] if idx != 0]
                target_tokens = [idx_to_token.get(idx, '<unk>') for idx in target_seq.cpu().numpy() if idx != 0]
                
                pred_tokens = pred_tokens[:generation_length]
                target_tokens = target_tokens[:generation_length]
                
                pred_text = ' '.join(pred_tokens)
                target_text = ' '.join(target_tokens)
                
                if pred_text and target_text:
                    scores = scorer.score(target_text, pred_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    
                    if len(examples) < 5:
                        examples.append((input_tokens[-5:], pred_tokens, target_tokens))
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2, examples