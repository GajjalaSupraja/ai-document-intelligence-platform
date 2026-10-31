import numpy as np

from app import embedding_service


class FakeSentenceTransformer:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, convert_to_numpy, normalize_embeddings):
        self.encoded_texts.append(texts)
        return np.ones((len(texts), 384), dtype="float32")


def test_embedding_service_uses_multilingual_e5_query_and_passage_prefixes(
    monkeypatch,
):
    fake_model = FakeSentenceTransformer()
    monkeypatch.setattr(
        embedding_service,
        "SentenceTransformer",
        lambda model_name: fake_model,
    )

    service = embedding_service.EmbeddingService()
    service.embed_texts(["A project report"])
    service.embed_query("ప్రాజెక్ట్ నివేదిక ఏమి చెబుతోంది?")

    assert embedding_service.MODEL_NAME == "intfloat/multilingual-e5-small"
    assert fake_model.encoded_texts == [
        ["passage: A project report"],
        ["query: ప్రాజెక్ట్ నివేదిక ఏమి చెబుతోంది?"],
    ]
