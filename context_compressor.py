"""
context_compressor.py — Context compression after reranking.

Trims each reranked document down to its most query-relevant sentences
before it goes into the generation prompt, cutting prompt token usage
without a round-trip to an LLM.

Deliberately NOT LLM-based: quest.py already makes 3 sequential Groq
calls per query (guardrail check, generation, faithfulness check).
Adding a 4th call here just to compress context would add latency and
cost that works against the entire point of "reduce unnecessary
tokens." A cheap lexical-overlap sentence scorer gets most of the
benefit (dropping clearly-irrelevant sentences from long chunks) for
effectively zero added latency.
"""

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "what", "how", "why", "when", "where", "who", "which", "do", "does",
}


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def compress_document_text(
    query: str,
    text: str,
    max_chars: int = 600,
    min_sentences_kept: int = 2,
) -> str:
    """
    Keep the sentences most relevant to `query` (by simple word-overlap
    scoring), in their original order, up to `max_chars`. If the whole
    text already fits under `max_chars`, it's returned unchanged — this
    only trims chunks that are actually long.
    """
    if len(text) <= max_chars:
        return text

    sentences = _split_sentences(text)
    if len(sentences) <= min_sentences_kept:
        # Too short to meaningfully compress — just hard-truncate.
        return text[:max_chars]

    query_terms = _tokenize(query)
    if not query_terms:
        return text[:max_chars]

    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_terms = _tokenize(sentence)
        overlap = len(query_terms & sentence_terms)
        scored.append((overlap, idx, sentence))

    # Highest-overlap sentences first, but keep at least
    # `min_sentences_kept` even if their overlap score is 0, so a chunk
    # never collapses to nothing.
    scored.sort(key=lambda t: t[0], reverse=True)

    kept_indices = set()
    total_chars = 0
    for overlap, idx, sentence in scored:
        if total_chars >= max_chars and len(kept_indices) >= min_sentences_kept:
            break
        kept_indices.add(idx)
        total_chars += len(sentence)

    # Re-assemble in original reading order, not score order — keeps the
    # compressed passage coherent instead of a jumbled sentence bag.
    kept_sentences = [sentences[i] for i in sorted(kept_indices)]
    return " ".join(kept_sentences)


def compress_documents(query: str, documents: list, max_chars: int = 600) -> list:
    """
    Returns a new list of Document-like objects with `.page_content`
    compressed. Metadata is preserved unchanged (including rerank_score,
    source_type, source_name, page, etc.) so citations.py keeps working
    exactly as before — only the text sent to the LLM shrinks.
    """
    compressed = []
    for doc in documents:
        new_text = compress_document_text(query, doc.page_content, max_chars=max_chars)
        # Mutate a shallow copy so we never touch the original Document
        # objects the caller (or FAISS's internal docstore) may still
        # reference elsewhere.
        doc_copy = doc.__class__(page_content=new_text, metadata=dict(doc.metadata))
        compressed.append(doc_copy)
    return compressed
