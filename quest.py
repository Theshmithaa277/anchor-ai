from dotenv import load_dotenv
import os

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from guardrails import check_query_safety
from reranker import rerank
from citations import format_context_with_citations, print_citations
from faithfulness import check_faithfulness
from query_rewriter import rewrite_query
from context_compressor import compress_documents

load_dotenv()

GENERATION_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

RAG_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. "
    "Answer only using the provided context. "
    "Do not use outside knowledge. "
    "If the answer is not present in the context, say "
    "'I don't know based on the provided context.' "
    "Keep the answer clear and structured using short paragraphs, "
    "bullet points, or numbered points when appropriate."
)


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in environment (.env)."
        )

    return Groq(api_key=api_key)


def get_vectorstore(index_path: str = FAISS_INDEX_PATH) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def run_query(
    query: str,
    client: Groq,
    vectorstore: FAISS,
    k_retrieve: int = 13,
    k_final: int = 4,
    verbose: bool = True,
) -> dict:

    result = {
        "query": query,
        "rewritten_query": query,
        "blocked": False,
        "answer": None,
        "citations": [],
        "faithfulness": None,
    }

    safety = check_query_safety(client, query)

    if not safety["safe"]:
        result["blocked"] = True
        result["answer"] = "I can't help with that request."
        return result

    # ---------------------------------------------------------
    # QUERY REWRITING (retrieval only — the ORIGINAL query is still
    # what gets answered below, so the response stays faithful to
    # exactly what the user asked).
    # ---------------------------------------------------------
    search_query = rewrite_query(client, query, verbose=verbose)
    result["rewritten_query"] = search_query

    raw_results = vectorstore.similarity_search(
        search_query,
        k=k_retrieve
    )

    if verbose:
        print(
            f"\nRetrieved {len(raw_results)} candidate chunks, "
            f"re-ranking to top {k_final}..."
        )

    top_docs = rerank(
        query,
        raw_results,
        top_n=k_final
    )

    # ---------------------------------------------------------
    # CONTEXT COMPRESSION (trim each reranked chunk to its most
    # query-relevant sentences before it goes into the prompt).
    # ---------------------------------------------------------
    top_docs = compress_documents(query, top_docs, max_chars=600)

    context, citation_list = format_context_with_citations(top_docs)
    result["citations"] = citation_list

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    result["answer"] = answer

    faithfulness = check_faithfulness(
        client,
        context,
        answer
    )

    result["faithfulness"] = faithfulness

    if verbose:
        print("\n===== GENERATED ANSWER =====\n")
        print(answer)
        print_citations(citation_list)

        if faithfulness["grounded"]:
            print(
                f"✅ Grounded "
                f"(confidence: {faithfulness['confidence']})"
            )
        else:
            print(
                f"⚠ POSSIBLY UNGROUNDED "
                f"(confidence: {faithfulness['confidence']})"
            )

    return result


if __name__ == "__main__":
    client = get_client()
    vectorstore = get_vectorstore()

    print("✅ Groq API key loaded successfully!")

    run_query(
        "What is the video about?",
        client=client,
        vectorstore=vectorstore,
        k_final=8
    )