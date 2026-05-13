import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.pkl"


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def create_vector_store(chunks: list, persist_dir: str = "./vector_store"):
    """Create FAISS index from chunks and save to disk."""
    os.makedirs(persist_dir, exist_ok=True)
    model = get_embedding_model()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    index_data = {
        "embeddings": embeddings,
        "texts": texts,
        "metadatas": metadatas,
    }

    index_path = os.path.join(persist_dir, INDEX_FILE)
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"Stored {len(texts)} chunks in '{index_path}'")
    return index_data


def load_vector_store(persist_dir: str = "./vector_store"):
    """Load existing FAISS index."""
    index_path = os.path.join(persist_dir, INDEX_FILE)
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    return index_data
