from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm

def evaluate_transformer():
    """
    Оцениваем предварительно подготовленную модел transformer
    """
    generator = pipeline("text-generation", model="distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    test_df = pd.read_csv('/home/ubuntu/text-autocomplete/data/test.csv')
    test_df['tokens'] = test_df['tokens'].apply(eval)
    test_df['text'] = test_df['cleaned_text']
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    for idx, row in tqdm(test_df.head(100).iterrows(), total=100, desc="Evaluating Transformer"):
        text = row['text']
        tokens = row['tokens']
        
        if len(tokens) < 10:
            continue
        
        split_point = int(len(tokens) * 0.75)
        input_tokens = tokens[:split_point]
        target_tokens = tokens[split_point:]
        
        input_text = ' '.join(input_tokens)
        target_text = ' '.join(target_tokens)
        
        try:
            result = generator(
                input_text,
                max_length=len(input_tokens) + len(target_tokens) + 1,
                do_sample=True,
                top_k=50,
                num_return_sequences=1
            )
            
            generated_text = result[0]['generated_text']
            
            generated_part = generated_text[len(input_text):].strip()
            
            scores = scorer.score(target_text, generated_part)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            
            if len(examples) < 5:
                examples.append((input_tokens, generated_part.split()[:10], target_tokens))
                
        except Exception as e:
            print(f"Error generating text: {e}")
            continue
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    print(f"Transformer Results:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print("\nExamples:")
    for i, (input_tokens, pred_tokens, target_tokens) in enumerate(examples):
        print(f"Example {i+1}:")
        print(f"  Input: {' '.join(input_tokens[-5:])}")
        print(f"  Pred: {' '.join(pred_tokens)}")
        print(f"  Target: {' '.join(target_tokens[:10])}")
        print()
    
    return avg_rouge1, avg_rouge2, examples

if __name__ == "__main__":
    evaluate_transformer()