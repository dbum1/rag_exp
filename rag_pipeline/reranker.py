# rag_pipeline/reranker.py

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

def rerank_cosine(query_vec, docs, embedder):
    doc_vecs = embedder.embed_documents([doc["content"] for doc in docs])
    scores = cosine_similarity([query_vec], doc_vecs)[0]
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in reranked]

def rerank_crossencoder(query, docs, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, doc["content"]) for doc in docs]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in reranked]

def rerank_documents(query, docs, method="none", query_vec=None, embedder=None, model_name=None):
    """
    method: 'none' | 'cosine' | 'crossencoder'
    query: 사용자의 질의 (text)
    docs: 검색된 문서 리스트
    query_vec: cosine rerank용 query vector
    embedder: cosine rerank용 embedder
    model_name: cross-encoder 모델명 (optional)
    """

    if method == "none":
        return docs

    if method == "cosine":
        if query_vec is None or embedder is None:
            raise ValueError("Cosine reranking requires query_vec and embedder.")
        return rerank_cosine(query_vec, docs, embedder)

    if method == "crossencoder":
        return rerank_crossencoder(query, docs, model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    raise ValueError(f"Unsupported rerank method: {method}")