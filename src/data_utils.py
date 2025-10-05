import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

nltk.download('punkt')

def load_and_clean_data(file_path):
    """
    Загрузка и очиститка набора данных
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts = f.readlines()
    
    df = pd.DataFrame({'text': texts})
    
    df = df[df['text'].str.strip().astype(bool)]
    
    return df

def clean_text(text):
    """
    Очистка и нормализация текста
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_texts(texts):
    """
    Токенизация списка текстов
    """
    tokenized_texts = []
    for text in texts:
        tokens = word_tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts

def prepare_dataset(input_file, output_dir):
    """
    Основная функция подготовки набора данных
    """
    df = load_and_clean_data(input_file)
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    df = df[df['cleaned_text'].str.len() > 10]
    
    tokenized_texts = tokenize_texts(df['cleaned_text'].tolist())
    df['tokens'] = tokenized_texts
    
    df = df[df['tokens'].apply(len) > 0]
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df[['cleaned_text', 'tokens']].to_csv(os.path.join(output_dir, 'dataset_processed.csv'), index=False)
    train_df[['cleaned_text', 'tokens']].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df[['cleaned_text', 'tokens']].to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df[['cleaned_text', 'tokens']].to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Dataset prepared:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    input_file = "/home/ubuntu/text-autocomplete/data/tweets.txt"
    output_dir = "/home/ubuntu/text-autocomplete/data"
    prepare_dataset(input_file, output_dir)