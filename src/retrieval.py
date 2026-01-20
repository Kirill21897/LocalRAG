def retrieve(query, q_client, embed_model, collection_name="docs_collection", top_k=5):
    # 1. Векторизуем запрос
    query_vector = embed_model.encode(query).tolist()

    # 2. Ищем в Qdrant
    search_results = q_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    # 3. Извлекаем тексты из payload
    retrieved_chunks = [
        {"text": hit.payload["text"], "score": hit.score, "metadata": hit.payload.get("metadata", {})}
        for hit in search_results.points
    ]

    print(f"Найдено {len(retrieved_chunks)} кандидатов через векторный поиск.")
    return retrieved_chunks