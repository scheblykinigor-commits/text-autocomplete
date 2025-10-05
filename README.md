# Нейросеть для автодополнения текстов

Проект по созданию модели автодополнения текста на основе LSTM и Transformers.

## Структура проекта
- `data/` - датасеты
│   ├── raw_dataset.csv "сырой" скачанный датасет
│   └── dataset_processed.csv "очищенный" датасет
│   ├── train.csv тренировочная выборка
│   ├── val.csv валидационная выборка
│   └── test.csv  тестовая выборка
- `src/` - исходный код
│   ├── data_utils.py Обработка датасета
|   ├── next_token_dataset.py код с torch Dataset'ом 
│   ├── lstm_model.py код lstm модели
|   ├── eval_lstm.py замер метрик lstm модели
|   ├── lstm_train.py код обучения модели
|   ├── eval_transformer_pipeline.py код с запуском и замером качества трансформера
- `configs/` - конфигурационные файлы
- `models/` - веса обученных моделей