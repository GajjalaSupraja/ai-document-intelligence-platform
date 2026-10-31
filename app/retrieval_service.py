
from app.embedding_service import EmbeddingService, VectorSearch


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.chunks: list[dict] = []
        self.vector_search = None

    def index_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            raise ValueError("No text chunks provided for indexing.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.vector_search = VectorSearch(embeddings)
        self.chunks = chunks

        return len(self.chunks)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        if self.vector_search is None:
            raise ValueError("Index a document before searching.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = self.embedding_service.embed_query(question)
        results = self.vector_search.search(query_embedding, top_k)

        return [
            {
                "chunk_index": index,
                "text": self.chunks[index]["text"],
                "score": round(score, 4),
                "source": self.chunks[index].get("source", "unknown"),
            }
            for index, score in results
        ]
