
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype="float32")

        embeddings = self.model.encode(
            [f"passage: {text}" for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")


class VectorSearch:
    def __init__(self, embeddings: np.ndarray):
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query = np.asarray(query_embedding, dtype="float32")

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match index.")

        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal),
        )

        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
