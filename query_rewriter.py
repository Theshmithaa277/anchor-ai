"""
query_rewriter.py — Query rewriting before retrieval.

Turns a user's raw, possibly terse or ambiguous question into a more
retrieval-friendly search query (expanding abbreviations, making
implicit references explicit) using a fast Groq model call.

This is fail-OPEN, unlike guardrails.py (which fails closed for
safety): if the rewrite call errors out or returns something unusable,
we fall back to the original query rather than blocking the request.
A broken rewrite should degrade retrieval quality, never break the app.

The rewritten query is used for retrieval only — quest.py still passes
the user's ORIGINAL question to the generation step, so the final
answer responds to exactly what the user asked.
"""

# A small/fast model for this — it's a single-sentence rewrite task, not
# a generation task, so paying for the large generation model here would
# be wasted latency and cost.
REWRITE_MODEL = "llama-3.1-8b-instant"

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into a single, clear, self-contained "
    "search query optimized for retrieving relevant passages from a "
    "document/video knowledge base. "
    "Expand abbreviations and vague pronouns where possible. "
    "Do not answer the question. Do not add explanation. "
    "Output ONLY the rewritten query, nothing else. "
    "If the original query is already clear, return it unchanged."
)

MAX_REWRITE_LENGTH = 300  # sanity cap; a "rewrite" this long is not a query


def rewrite_query(client, query: str, verbose: bool = False) -> str:
    """
    Returns a retrieval-optimized version of `query`. On any failure,
    returns `query` unchanged.
    """
    query = (query or "").strip()
    if not query:
        return query

    try:
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=120,
        )

        rewritten = (response.choices[0].message.content or "").strip()

        # Strip a leading/trailing quote the model sometimes adds.
        rewritten = rewritten.strip('"').strip("'").strip()

        if not rewritten or len(rewritten) > MAX_REWRITE_LENGTH:
            if verbose:
                print("⚠ Query rewrite unusable, falling back to original query.")
            return query

        if verbose and rewritten.lower() != query.lower():
            print(f"ℹ Query rewritten: {query!r} -> {rewritten!r}")

        return rewritten

    except Exception as e:
        if verbose:
            print(f"⚠ Query rewrite failed ({e}), falling back to original query.")
        return query
