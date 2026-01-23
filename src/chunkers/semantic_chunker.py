from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List

def chunk_semantic(text: str, threshold: float = 0.6) -> List[str]:
    """
    Семантическое чанкирование.
    1. Разбивает текст на предложения.
    2. Вычисляет эмбеддинги для каждого предложения.
    3. Сравнивает косинусное сходство соседних предложений.
    4. Если сходство меньше порога (threshold), начинается новый чанк.
    """
    # 1. Разбивка на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences) if x.strip()]
    
    if not sentences:
        return []
        
    # 2. Инициализация модели (используем ту же, что и в проекте, если возможно, или легкую)
    # Используем легкую модель для скорости в демо
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 3. Эмбеддинги
    texts = [x['sentence'] for x in sentences]
    embeddings = model.encode(texts)
    
    # 4. Группировка
    chunks = []
    current_chunk_sentences = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]['sentence']
        current_chunk_sentences.append(sentence)
        
        # Если это последнее предложение, просто сохраняем чанк
        if i == len(sentences) - 1:
            chunks.append(" ".join(current_chunk_sentences))
            break
            
        # Считаем сходство с следующим предложением
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        
        # Если сходство ниже порога, значит тема сменилась -> закрываем чанк
        if sim < threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []
            
    return chunks
