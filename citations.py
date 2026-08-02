def format_context_with_citations(documents: list) -> tuple[str, list]:
    """
    Takes reranked Document objects, returns:
      - a context string with [Source N] tags baked in (for the LLM prompt)
      - a citation list (for displaying to the user afterward)
    """
    context_parts = []
    citation_list = []

    for i, doc in enumerate(documents, 1):
        source_type = doc.metadata.get("source_type", "unknown")
        source_name = doc.metadata.get("source_name", "unknown")

        context_parts.append(f"[Source {i} — {source_type}: {source_name}]\n{doc.page_content}")
        citation_list.append({
            "id": i,
            "source_type": source_type,
            "source_name": source_name,
            "rerank_score": doc.metadata.get("rerank_score"),
        })

    context = "\n\n".join(context_parts)
    return context, citation_list


def print_citations(citation_list: list):
    print("\n----- SOURCES -----")
    for c in citation_list:
        score_str = f" (relevance: {c['rerank_score']:.2f})" if c.get("rerank_score") is not None else ""
        print(f"[{c['id']}] {c['source_type']}: {c['source_name']}{score_str}")