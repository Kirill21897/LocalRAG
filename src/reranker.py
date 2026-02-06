from sentence_transformers import CrossEncoder

# Загружаем модель один раз при импорте
_rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, chunks, top_n=3):
    """
    Переранжирование списка чанков по релевантности к запросу.

    Args:
        query (str): Запрос пользователя.
        chunks (list): Список чанков, каждый с полями 'text', 'score', 'metadata'.
        top_n (int): Сколько чанков вернуть после реранжирования.

    Returns:
        list: Топ-N реранжированных чанков.
    """
    if not chunks:
        return []

    # Подготавливаем пары (запрос, текст чанка)
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # Получаем оценки релевантности
    scores = _rerank_model.predict(pairs, show_progress_bar=False)

    # Добавляем скоры к чанкам
    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)  # преобразуем в float для совместимости

    # Сортируем по убыванию оценки реранкера
    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # print(f"Переранжирование завершено. Выбрано топ-{top_n} наиболее релевантных.")
    return sorted_chunks[:top_n]