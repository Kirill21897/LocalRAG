# config\config.py
from pathlib import Path

# DATA
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# LLM & API CONFIG
OLLAMA_BASE_URL = "http://192.168.88.21:91/v1"
OLLAMA_API_KEY = "ollama"
JUDGE_MODEL = "gpt-oss:20b"  # Model used for evaluation
EMBEDDING_MODEL = "nomic-embed-text" # Ollama embedding model
