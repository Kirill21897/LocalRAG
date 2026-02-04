from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List

def chunk_markdown(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Разбивает Markdown текст по заголовкам с дополнительной рекурсивной разбивкой
    для длинных секций.
    
    Args:
        text (str): Входной Markdown текст.
        chunk_size (int): Максимальный размер чанка (для вторичной разбивки).
        chunk_overlap (int): Перекрытие (для вторичной разбивки).
    """
    # 1. Разбивка по заголовкам
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = md_splitter.split_text(text)
    
    # 2. Вторичная разбивка длинных секций
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []
    
    for doc in header_splits:
        # Формируем контекст из заголовков
        header_context = " > ".join(f"{v}" for k, v in doc.metadata.items())
        
        # Если секция слишком большая, разбиваем её дополнительно
        if len(doc.page_content) > chunk_size:
            sub_docs = text_splitter.create_documents([doc.page_content])
            for sub_doc in sub_docs:
                # Добавляем контекст заголовка к каждому под-чанку
                content = f"[{header_context}]\n{sub_doc.page_content}" if header_context else sub_doc.page_content
                final_chunks.append(content)
        else:
            content = f"[{header_context}]\n{doc.page_content}" if header_context else doc.page_content
            final_chunks.append(content)
        
    return final_chunks
