import json
from groq import Groq

FAITHFULNESS_MODEL = "openai/gpt-oss-120b"

FAITHFULNESS_PROMPT = """You are a strict fact-checker for a RAG system.

You will be given:
1. CONTEXT — the retrieved source chunks
2. ANSWER — an AI-generated answer that should be based ONLY on the context

Your job: determine if the ANSWER is fully supported by the CONTEXT.

Rules:
- If the answer contains ANY claim not present in or inferable from the context, mark it ungrounded.
- If the answer says "I don't know based on the provided context," that is always grounded (it's honest).
- Be strict: paraphrasing is fine, but invented facts, numbers, or examples not in the context are NOT fine.

Respond with ONLY this JSON format, no other text:
{"grounded": true or false, "confidence": "high"/"medium"/"low", "unsupported_claims": ["list of specific claims not found in context, empty if none"]}
"""


def check_faithfulness(client: Groq, context: str, answer: str) -> dict:
    """
    Check whether an answer is grounded in the provided context.

    Fail-closed by design: if the checker's API call fails, or its output
    doesn't parse, we report the answer as UNGROUNDED rather than grounded.
    This is a hallucination-detection component — silently marking an
    unverifiable answer as "trustworthy" defeats its purpose.

    Returns: {"grounded": bool, "confidence": str, "unsupported_claims": list}
    """
    if "i don't know based on the provided context" in answer.lower():
        return {"grounded": True, "confidence": "high", "unsupported_claims": []}

    try:
        response = client.chat.completions.create(
            model=FAITHFULNESS_MODEL,
            messages=[
                {"role": "system", "content": FAITHFULNESS_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
            ],
            temperature=0.0,
            max_tokens=800,
            reasoning_effort="low",
        )
    except Exception as e:
        print(f"⚠ Faithfulness checker API call failed: {e}")
        return {"grounded": False, "confidence": "low", "unsupported_claims": ["api_call_failed"]}

    raw_output = response.choices[0].message.content.strip()

    try:
        cleaned = raw_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return {
            "grounded": parsed.get("grounded", False),
            "confidence": parsed.get("confidence", "medium"),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except (json.JSONDecodeError, AttributeError):
        print(f"⚠ Could not parse faithfulness checker output: {raw_output}")
        return {"grounded": False, "confidence": "low", "unsupported_claims": ["parse_error"]}