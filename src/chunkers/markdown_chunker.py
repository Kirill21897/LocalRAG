from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List

def chunk_markdown(text: str) -> List[str]:
    """
    Разбивает Markdown текст по заголовкам.
    Это позволяет сохранить семантическую структуру документа.
    Возвращает список строк, где каждая строка - это контент секции вместе с метаданными (в виде текста).
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(text)
    
    # Собираем обратно в строки для простоты использования в пайплайне, 
    # добавляя контекст заголовка к тексту
    result_chunks = []
    for doc in docs:
        header_context = " > ".join(f"{k}: {v}" for k, v in doc.metadata.items())
        content = f"[{header_context}]\n{doc.page_content}" if header_context else doc.page_content
        result_chunks.append(content)
        
    return result_chunks
