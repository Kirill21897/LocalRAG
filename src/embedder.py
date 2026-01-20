import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def vectorize_and_upload(chunks, file_name, collection_name="docs_collection"):
    # 1. Инициализация модели и клиента
    # Для работы в памяти используйте ":memory:", для Docker — "http://localhost:6333"
    client = QdrantClient(":memory:") 
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    vector_size = 384 # Размерность для данной модели

    # 2. Создаем или пересоздаем коллекцию
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # 3. Генерация эмбеддингов
    print(f"Генерация векторов для {len(chunks)} чанков...")
    embeddings = model.encode(chunks)

    # 4. Формирование точек (Points) для Qdrant
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk,
                    "metadata": {
                        "source": file_name,
                        "chunk_id": i
                    }
                }
            )
        )

    # 5. Загрузка в базу
    client.upsert(collection_name=collection_name, points=points)
    print(f"Данные успешно загружены в коллекцию '{collection_name}'")
    
    return client, model # Возвращаем для дальнейшего поиска