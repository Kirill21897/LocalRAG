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
    # Список сокращений, которые не должны считаться концом предложения
    abbreviations = [
        "г.", "ул.", "им.", "см.", "т.д.", "т.п.", "т.е.", "руб.", "коп.", "кв.", "м.", "кг.",
        "рис.", "табл.", "стр.", "просп.", "пер.", "обл."
    ]
    
    # Временная замена точек в сокращениях на спецсимвол
    processed_text = text
    for abbr in abbreviations:
        processed_text = processed_text.replace(abbr, abbr.replace(".", "<DOT>"))
        
    # Разбивка на предложения по точке, вопросу или восклицательному знаку
    # Используем более сложный regex, чтобы захватить разделитель
    sentences = re.split(r'(?<=[.!?])\s+', processed_text)
    
    # Восстановление точек и очистка
    clean_sentences = []
    for s in sentences:
        s_restored = s.replace("<DOT>", ".")
        if s_restored.strip():
            clean_sentences.append(s_restored.strip())
            
    sentences = clean_sentences
    
    chunks = []
    if not sentences:
        return chunks
        
    for i in range(len(sentences)):
        # Определяем границы окна
        # window_size=3 -> 1 предл до, текущее, 1 предл после
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(len(sentences), i + half_window + 1)
        
        # Собираем окно
        window_group = sentences[start:end]
        chunk = " ".join(window_group)
        chunks.append(chunk)
        
    return chunks
