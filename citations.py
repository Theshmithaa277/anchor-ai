def format_context_with_citations(documents: list) -> tuple[str, list]:
    """
    Takes reranked Document objects, returns:
      - a context string with [Source N] tags baked in (for the LLM prompt)
      - a citation list (for displaying to the user afterward)

    Metadata-aware: also surfaces page number (PDFs, via PyPDFLoader's
    built-in "page" field) and ingestion timestamp when present, so a
    citation can say more than just "which file" — it can say "which
    page, ingested when." Fields are omitted (not set to a placeholder)
    when a given source type doesn't have them, e.g. YouTube transcripts
    have no page number.
    """
    context_parts = []
    citation_list = []

    for i, doc in enumerate(documents, 1):
        source_type = doc.metadata.get("source_type", "unknown")
        source_name = doc.metadata.get("source_name", "unknown")
        page = doc.metadata.get("page")
        ingested_at = doc.metadata.get("ingested_at")

        label = f"[Source {i} — {source_type}: {source_name}"
        if page is not None:
            label += f", page {page + 1}"  # PyPDFLoader's "page" is 0-indexed
        label += "]"

        context_parts.append(f"{label}\n{doc.page_content}")

        citation = {
            "id": i,
            "source_type": source_type,
            "source_name": source_name,
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        if page is not None:
            citation["page_number"] = page + 1
        if ingested_at is not None:
            citation["ingested_at"] = ingested_at

        citation_list.append(citation)

    context = "\n\n".join(context_parts)
    return context, citation_list


def print_citations(citation_list: list):
    print("\n----- SOURCES -----")
    for c in citation_list:
        score_str = f" (relevance: {c['rerank_score']:.2f})" if c.get("rerank_score") is not None else ""
        page_str = f", page {c['page_number']}" if c.get("page_number") is not None else ""
        print(f"[{c['id']}] {c['source_type']}: {c['source_name']}{page_str}{score_str}")