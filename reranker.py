import copy
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_model = None


def get_reranker_model() -> CrossEncoder:
    """
    Lazily load and cache the cross-encoder model.
    Call this (instead of relying on a module-level global) so a host app
    like Streamlit can wrap it with @st.cache_resource if desired.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def rerank(query: str, documents: list, top_n: int = 4) -> list:
    """
    documents: list of LangChain Document objects (from FAISS similarity_search)
    Returns: top_n documents (copies), re-ordered by cross-encoder relevance score.
    """
    if not documents:
        return []

    model = get_reranker_model()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = []
    for doc, score in scored_docs[:top_n]:
        # Copy instead of mutating the original Document in place — avoids
        # side effects if the same Document objects are reused/cached elsewhere.
        doc_copy = copy.deepcopy(doc)
        doc_copy.metadata["rerank_score"] = float(score)
        top_docs.append(doc_copy)

    return top_docs