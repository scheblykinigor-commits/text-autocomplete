import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

nltk.download('punkt')

def download_dataset(url="https://code.s3.yandex.net/deep-learning/tweets.txt", save_path="../data/raw_dataset.csv"):
    """Скачивание датасета"""
    try:
        # Пробуем разные способы загрузки
        df = pd.read_csv(url, encoding='latin-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'], 
                        quoting=3, error_bad_lines=False)
    except:
        try:
            # Альтернативный способ загрузки
            df = pd.read_csv(url, encoding='latin-1', header=None, 
                           names=['target', 'id', 'date', 'flag', 'user', 'text'],
                           on_bad_lines='skip')
        except:
            # Самый простой способ - загрузить как текст и разобрать вручную
            df = pd.read_csv(url, encoding='latin-1', header=None)
            # Оставляем только первые 6 колонок
            df = df.iloc[:, :6]
            df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Сохраняем сырые данные
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def clean_text(text):
    """Очистка и нормализация текста"""
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление ссылок
    text = re.sub(r'http\S+', '', text)
    
    # Удаление упоминаний
    text = re.sub(r'@\w+', '', text)
    
    # Удаление хэштегов
    text = re.sub(r'#\w+', '', text)
    
    # Удаление специальных символов, оставляем только буквы, цифры и основные знаки препинания
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s\.\,\!\?\:]', '', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Токенизация текста"""
    return word_tokenize(text)

def prepare_dataset(raw_data_path="../data/raw_dataset.csv", processed_data_path="../data/dataset_processed.csv"):
    """Основная функция подготовки данных"""
    
    # Загрузка данных
    df = pd.read_csv(raw_data_path)
    
    # Очистка текста
    print("Очистка текста...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Токенизация
    print("Токенизация...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    
    # Удаление пустых текстов
    df = df[df['tokens'].apply(len) > 0]
    
    # Сохранение обработанного датасета
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    
    print(f"Обработано {len(df)} записей")
    return df

def split_dataset(processed_data_path="../data/dataset_processed.csv"):
    """Разделение датасета на train/val/test"""
    
    df = pd.read_csv(processed_data_path)
    
    # Преобразование строки токенов обратно в список
    df['tokens'] = df['tokens'].apply(eval)
    
    # Разделение на train (80%), temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Разделение temp на val (10%) и test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Сохранение разделенных датасетов
    train_df.to_csv("../data/train.csv", index=False)
    val_df.to_csv("../data/val.csv", index=False)
    test_df.to_csv("../data/test.csv", index=False)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df