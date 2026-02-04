from langchain_text_splitters import TokenTextSplitter
from typing import List

def chunk_token(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки, основываясь на токенах (cl100k_base для OpenAI или стандартный tiktoken).
    
    Args:
        text (str): Входной текст.
        chunk_size (int): Размер чанка в токенах.
        chunk_overlap (int): Перекрытие в токенах.
    """
    # Используем модель 'gpt-4' (encoding cl100k_base) для более точного подсчета токенов
    # для современных моделей, или fallback на дефолт.
    try:
        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",  # Явно указываем энкодинг для согласованности
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    except Exception as e:
        print(f"Warning: Failed to initialize TokenTextSplitter with cl100k_base: {e}. Falling back to default encoding.")
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]
