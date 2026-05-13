import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rag.embeddings import load_vector_store, EMBEDDING_MODEL


class HybridRetriever:
    """Combines semantic search (cosine similarity) with keyword search (BM25)."""

    def __init__(self, persist_dir: str = "./vector_store", semantic_weight: float = 0.7):
        index_data = load_vector_store(persist_dir)
        self.embeddings = index_data["embeddings"]
        self.doc_texts = index_data["texts"]
        self.doc_metadatas = index_data["metadatas"]
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        tokenized = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Hybrid retrieval: semantic + keyword, returns top_k results with scores."""

        # Semantic search via cosine similarity
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_normalized = bm25_scores / max_bm25

        # Combine scores
        combined_scores = (
            self.semantic_weight * similarities
            + self.keyword_weight * bm25_normalized
        )

        # Get top_k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": f"chunk_{idx}",
                "text": self.doc_texts[idx],
                "metadata": self.doc_metadatas[idx],
                "score": float(combined_scores[idx]),
                "semantic_score": float(similarities[idx]),
                "bm25_score": float(bm25_normalized[idx]),
            })

        return results
