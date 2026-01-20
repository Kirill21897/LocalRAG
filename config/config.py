# config\config.py
from pathlib import Path
from sentence_transformers import SentenceTransformer

# DATA
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"