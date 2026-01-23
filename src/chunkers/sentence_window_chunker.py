import re
from typing import List

def chunk_sentence_window(text: str, window_size: int = 3) -> List[str]:
    """
    Разбивает текст на предложения и формирует чанки методом скользящего окна.
    Каждый чанк содержит центральное предложение и (window_size - 1)/2 предложений вокруг.
    
    Args:
        text (str): Входной текст.
        window_size (int): Размер окна (количество предложений в одном чанке).
    """
    # Простая разбивка на предложения по точке, вопросу или восклицательному знаку.
    # В продакшене лучше использовать nltk или spacy.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    if not sentences:
        return chunks
        
    for i in range(len(sentences)):
        # Определяем границы окна
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        
        # Собираем окно
        window_group = sentences[start:end]
        chunk = " ".join(window_group)
        chunks.append(chunk)
        
    return chunks
