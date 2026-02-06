from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List, Union
import config.config as cfg
import numpy as np

_SHARED_EMBEDDING_MODEL = None

class OllamaEmbeddingWrapper(Embeddings):
    """
    Wrapper for OllamaEmbeddings that provides a SentenceTransformer-compatible interface (encode method)
    AND LangChain Embeddings interface (embed_documents, embed_query).
    """
    def __init__(self, model_name: str, base_url: str):
        # Clean base_url (remove /v1 if present)
        ollama_url = base_url.replace("/v1", "")
        self.client = OllamaEmbeddings(
            model=model_name,
            base_url=ollama_url
        )
        self.model_name = model_name

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> Union[List[float], np.ndarray]:
        """
        Mimics SentenceTransformer.encode
        """
        if isinstance(sentences, str):
            # Single string -> return single vector (as list or numpy array)
            # SentenceTransformer returns numpy array by default, but list is often fine.
            # To be safe and compatible with Qdrant which expects list of floats, we return numpy array.
            vec = self.client.embed_query(sentences)
            return np.array(vec)
        else:
            # List of strings -> return list of vectors
            vecs = self.client.embed_documents(sentences)
            return np.array(vecs)

    # LangChain Embeddings Interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)


def get_shared_embedding_model(model_name=None):
    """
    Returns a singleton instance of the OllamaEmbeddingWrapper.
    """
    global _SHARED_EMBEDDING_MODEL
    if model_name is None:
        model_name = cfg.EMBEDDING_MODEL

    if _SHARED_EMBEDDING_MODEL is None:
        # print(f"Loading shared embedding model (Ollama): {model_name}...")
        _SHARED_EMBEDDING_MODEL = OllamaEmbeddingWrapper(
            model_name=model_name,
            base_url=cfg.OLLAMA_BASE_URL
        )
    return _SHARED_EMBEDDING_MODEL

class SharedEmbeddings(Embeddings):
    """
    LangChain compatible wrapper for the shared embedding model.
    Just delegates to the wrapper instance.
    """
    def __init__(self):
        self.model = get_shared_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
