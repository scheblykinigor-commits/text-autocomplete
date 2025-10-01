from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm

def evaluate_transformer(test_data_path, num_examples=100):
    """Оценка предобученной модели трансформера"""
    
    # Загрузка модели и токенизатора
    generator = pipeline("text-generation", model="distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Загрузка тестовых данных
    df = pd.read_csv(test_data_path)
    df = df.head(num_examples)  # Ограничиваем количество примеров для скорости
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Transformer"):
        text = row['cleaned_text']
        tokens = eval(row['tokens'])
        
        if len(tokens) < 4:
            continue
            
        # Разделение на вход (3/4) и цель (1/4)
        split_point = int(len(tokens) * 0.75)
        input_tokens = tokens[:split_point]
        target_tokens = tokens[split_point:]
        
        input_text = ' '.join(input_tokens)
        target_text = ' '.join(target_tokens)
        
        try:
            # Генерация продолжения
            result = generator(input_text, max_length=len(tokens) + 5, do_sample=True, top_k=50)
            generated_text = result[0]["generated_text"]
            
            # Извлечение сгенерированной части (убираем исходный текст)
            generated_continuation = generated_text[len(input_text):].strip()
            
            # Вычисление ROUGE
            scores = scorer.score(target_text, generated_continuation)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            
            # Показ примеров
            if idx < 5:
                print(f"Input: {input_text}")
                print(f"Target: {target_text}")
                print(f"Generated: {generated_continuation}")
                print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Вычисление средних метрик
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    print(f"Transformer Results - ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}")
    
    return avg_rouge1, avg_rouge2