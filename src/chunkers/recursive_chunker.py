from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def chunk_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Разбивает текст на чанки рекурсивно, пытаясь сохранить структуру параграфов и предложений.
    Использует стандартные разделители LangChain ["\n\n", "\n", " ", ""].
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]
