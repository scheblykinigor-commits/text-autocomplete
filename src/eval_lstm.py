import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def calculate_rouge(model, dataloader, device, vocab, num_examples=5):
    """Вычисление метрик ROUGE для LSTM модели"""
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    examples_shown = 0
    
    # Создание обратного словаря для декодирования
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Генерация предсказаний
            output, _ = model(input_ids)
            predictions = torch.argmax(output, dim=-1)
            
            # Обработка каждого примера в батче
            for i in range(len(input_ids)):
                # Декодирование исходного текста
                input_tokens = [reverse_vocab[idx.item()] for idx in input_ids[i] if idx.item() != 0]
                target_tokens = [reverse_vocab[idx.item()] for idx in target_ids[i] if idx.item() != 0]
                pred_tokens = [reverse_vocab[idx.item()] for idx in predictions[i] if idx.item() != 0]
                
                # Удаление паддинга
                input_tokens = [t for t in input_tokens if t != '<PAD>']
                target_tokens = [t for t in target_tokens if t != '<PAD>']
                pred_tokens = [t for t in pred_tokens if t != '<PAD>']
                
                # Создание строк для вычисления ROUGE
                reference = ' '.join(target_tokens)
                prediction = ' '.join(pred_tokens)
                
                if reference and prediction:
                    scores = scorer.score(reference, prediction)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                
                # Показ примеров
                if examples_shown < num_examples and input_tokens and target_tokens and pred_tokens:
                    print(f"Input: {' '.join(input_tokens)}")
                    print(f"Target: {' '.join(target_tokens)}")
                    print(f"Predicted: {' '.join(pred_tokens)}")
                    print("-" * 50)
                    examples_shown += 1
                
                if examples_shown >= num_examples:
                    break
            
            if examples_shown >= num_examples:
                break
    
    # Вычисление средних метрик
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2