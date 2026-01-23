from langchain_text_splitters import TokenTextSplitter
from typing import List

def chunk_token(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки, основываясь на токенах (приблизительно соответствует токенам LLM).
    Это полезно для строгого соблюдения ограничений контекстного окна модели.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]
