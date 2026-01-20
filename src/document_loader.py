# src/document_loader.py
from docling.document_converter import DocumentConverter
import os
from pathlib import Path
from config.config import PROCESSED_DATA_DIR

def load_document(file_path: str, save_intermediate: bool = True) -> str:
    """
    Загружает документ, конвертирует его в Markdown, опционально сохраняет 
    результат в .md файл и возвращает строку Markdown.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    md_content = result.document.export_to_markdown()
    
    if save_intermediate:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        input_filename = Path(file_path).stem
        save_path = Path(PROCESSED_DATA_DIR) / f"{input_filename}.md"
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print(f"Saved intermediate file: {save_path}")
    
    return md_content