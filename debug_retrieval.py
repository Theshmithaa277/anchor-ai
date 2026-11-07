"""
debug_retrieval.py — READ-ONLY diagnostic for Anchor AI retrieval.

It loads the saved FAISS index from disk and reports what is in it. It never
modifies the index, the app, or any project file.

Usage (run in a SECOND terminal while the app is still running — do NOT
restart app.py first, because the app wipes the index on startup):

    python debug_retrieval.py
        -> what is stored, per source (PDFs, images, videos)

    python debug_retrieval.py "your question here"
        -> also shows how that question ranks, scores and filters chunks,
           mirroring the real pipeline in quest.py

    Optional flag:  --no-rewrite   skip the query-rewriting API call
"""

import inspect
import os
import sys
from collections import OrderedDict, defaultdict

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import load_to_faiss
import quest
from reranker import rerank

INDEX_PATH = "faiss_index"


def shorten(text: str, n: int = 90) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def load_index():
    if not os.path.isdir(INDEX_PATH):
        print("\nNo saved index found (folder 'faiss_index' does not exist).")
        print("The app wipes the index every time it starts. To use this tool:")
        print("  1. Start the app and upload your files (PDFs + the video).")
        print("  2. WITHOUT restarting the app, open a second terminal and run this script.")
        sys.exit(0)

    embeddings = HuggingFaceEmbeddings(model_name=load_to_faiss.EMBEDDING_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def print_inventory(vectorstore):
    docs = list(vectorstore.docstore._dict.values())
    print("=" * 78)
    print(f"INDEX CONTENTS: {len(docs)} chunks in total")
    print("=" * 78)

    groups = OrderedDict()
    for d in docs:
        key = (d.metadata.get("source_type", "unknown"), d.metadata.get("source_name", "unknown"))
        groups.setdefault(key, []).append(d)

    for (stype, sname), chunks in groups.items():
        chars = sum(len(c.page_content) for c in chunks)
        stamp = chunks[0].metadata.get("ingested_at", "n/a")
        print(f"\n[{stype}] {shorten(sname, 70)}")
        print(f"    chunks: {len(chunks)} | characters: {chars} | ingested_at: {stamp}")
        print(f"    first chunk starts: {shorten(chunks[0].page_content, 100)}")

    types_present = {k[0] for k in groups}
    print("\nSources by type:", {t: sum(1 for k in groups if k[0] == t) for t in sorted(types_present)})
    if "youtube" not in types_present:
        print("\n>>> No YouTube video is stored in the index. If you uploaded one, it never")
        print("    reached the index (or the app was restarted afterwards).")
    return docs


def print_query_report(vectorstore, query: str, use_rewrite: bool):
    k_retrieve = inspect.signature(quest.run_query).parameters["k_retrieve"].default
    k_final = inspect.signature(quest.run_query).parameters["k_final"].default
    floor = quest.MIN_RERANK_SCORE

    print("\n" + "=" * 78)
    print(f"QUESTION: {query}")
    print("=" * 78)

    search_query = query
    if use_rewrite:
        try:
            client = quest.get_client()
            search_query = quest.rewrite_query(client, query, verbose=False)
        except Exception as e:
            print(f"(query rewriting unavailable: {e} — using your question as-is)")
    print(f"Search query used for retrieval: {search_query!r}")
    print(f"Pipeline settings: candidates={k_retrieve}, kept after rerank={k_final}, relevance floor={floor}")

    # --- Stage 1: FAISS similarity search (same call the pipeline makes) ---
    hits = vectorstore.similarity_search_with_score(search_query, k=k_retrieve)
    print(f"\nSTAGE 1 — FAISS returned {len(hits)} candidate chunks (lower distance = closer):")
    print(f"{'rank':>4}  {'dist':>6}  {'type':<8} {'source':<34} {'pg':>3}  text")
    for i, (doc, dist) in enumerate(hits, 1):
        pg = doc.metadata.get("page")
        print(f"{i:>4}  {dist:>6.3f}  {doc.metadata.get('source_type','?'):<8} "
              f"{shorten(doc.metadata.get('source_name','?'), 34):<34} "
              f"{(pg + 1) if pg is not None else '-':>3}  {shorten(doc.page_content, 60)}")

    by_type = defaultdict(list)
    for i, (doc, _) in enumerate(hits, 1):
        by_type[doc.metadata.get("source_type", "?")].append(i)
    print("\nWhere each source type landed in the candidate list:")
    for t, ranks in sorted(by_type.items()):
        print(f"   {t:<8} {len(ranks)} candidate(s), best rank #{min(ranks)}")
    all_types = {d.metadata.get("source_type", "?") for d in vectorstore.docstore._dict.values()}
    for t in sorted(all_types - set(by_type)):
        print(f"   {t:<8} 0 candidates — NONE of its chunks were retrieved for this question")

    # --- Stage 2: cross-encoder rerank (pipeline reranks with the ORIGINAL question) ---
    candidates = [d for d, _ in hits]
    reranked = rerank(query, candidates, top_n=len(candidates))
    print(f"\nSTAGE 2 — reranker scores (pipeline keeps the top {k_final}; "
          f"chunks scoring below {floor} are dropped):")
    print(f"{'#':>3}  {'score':>7}  {'type':<8} {'source':<34} text")
    for i, doc in enumerate(reranked[:10], 1):
        s = doc.metadata.get("rerank_score", 0)
        mark = "" if i <= k_final else "   (cut: outside top " + str(k_final) + ")"
        print(f"{i:>3}  {s:>7.2f}  {doc.metadata.get('source_type','?'):<8} "
              f"{shorten(doc.metadata.get('source_name','?'), 34):<34} "
              f"{shorten(doc.page_content, 50)}{mark}")

    # --- Stage 3: relevance floor, exactly as quest.run_query does it ---
    top = reranked[:k_final]
    if len(top) > 1:
        strong = [d for d in top if d.metadata.get("rerank_score", 0) >= floor]
        sent = strong if strong else top[:1]
    else:
        sent = top

    print(f"\nSTAGE 3 — chunks that would actually be sent to the LLM ({len(sent)}):")
    for d in sent:
        print(f"   [{d.metadata.get('source_type','?')}] {shorten(d.metadata.get('source_name','?'), 40)}"
              f"  score={d.metadata.get('rerank_score', 0):.2f}")
    dropped = [d for d in top if d not in sent]
    for d in dropped:
        print(f"   DROPPED by relevance floor: [{d.metadata.get('source_type','?')}] "
              f"{shorten(d.metadata.get('source_name','?'), 30)} score={d.metadata.get('rerank_score', 0):.2f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_rewrite = "--no-rewrite" not in sys.argv

    vectorstore = load_index()
    print_inventory(vectorstore)
    if args:
        print_query_report(vectorstore, " ".join(args), use_rewrite)
    else:
        print("\nTip: add a question to see how it is ranked, e.g.")
        print('    python debug_retrieval.py "what is the video about?"')


if __name__ == "__main__":
    main()