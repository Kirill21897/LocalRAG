try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    raise ImportError(f"Semantic chunker requires extra dependencies. Please install them: pip install sentence-transformers scikit-learn numpy. Error: {e}")

import re
from typing import List
from src.model_cache import get_shared_embedding_model

def chunk_semantic(text: str, threshold: float = 0.6) -> List[str]:
    """
    Семантическое чанкирование.
    1. Разбивает текст на предложения.
    2. Вычисляет эмбеддинги для каждого предложения.
    3. Сравнивает косинусное сходство соседних предложений.
    4. Если сходство меньше порога (threshold), начинается новый чанк.
    """
    # 1. Разбивка на предложения (используем ту же логику, что и в sentence window для консистентности)
    abbreviations = [
        "г.", "ул.", "им.", "см.", "т.д.", "т.п.", "т.е.", "руб.", "коп.", "кв.", "м.", "кг.",
        "рис.", "табл.", "стр.", "просп.", "пер.", "обл."
    ]
    processed_text = text
    for abbr in abbreviations:
        processed_text = processed_text.replace(abbr, abbr.replace(".", "<DOT>"))
        
    raw_sentences = re.split(r'(?<=[.!?])\s+', processed_text)
    
    sentences = []
    for s in raw_sentences:
        s_restored = s.replace("<DOT>", ".").strip()
        if s_restored:
            sentences.append({'sentence': s_restored})
            
    if not sentences:
        return []
        
    # 2. Получение модели и эмбеддингов
    model = get_shared_embedding_model()
    
    texts = [x['sentence'] for x in sentences]
    embeddings = model.encode(texts)
    
    # 3. Группировка
    chunks = []
    current_chunk_sentences = []
    
    # Добавляем первое предложение
    current_chunk_sentences.append(sentences[0]['sentence'])
    
    for i in range(len(sentences) - 1):
        # Считаем сходство между текущим и следующим предложением
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        
        if sim >= threshold:
            # Тема продолжается
            current_chunk_sentences.append(sentences[i+1]['sentence'])
        else:
            # Тема сменилась, закрываем текущий чанк
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i+1]['sentence']]
            
    # Не забываем добавить последний чанк, если он не пуст
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
            
    return chunks
