def chunk_text(text, chunk_size, overlap):
    # Проверка на логическую ошибку
    if overlap >= chunk_size:
        raise ValueError("Перекрытие (overlap) должно быть меньше размера чанка (chunk_size).")

    chunks = []
    start = 0
    
    while start < len(text):
        # Берем кусок текста заданной длины
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Сдвигаем указатель начала: размер чанка минус перекрытие
        start += (chunk_size - overlap)
        
    return chunks