import json
from quest import run_query, get_client, get_vectorstore

REFUSAL_PHRASE = "i don't know based on the provided context"


def load_test_set(path: str = "eval_data/test_questions.json") -> list:
    with open(path, "r") as f:
        return json.load(f)


def run_eval(test_set: list) -> dict:
    # run_query now needs client/vectorstore passed in explicitly (see quest.py) —
    # load them once here and reuse across the whole eval run.
    client = get_client()
    vectorstore = get_vectorstore()

    results = []
    retrieval_hits = 0
    retrieval_total = 0

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    # Refusals ("I don't know based on the provided context") are tracked
    # separately from the hallucination confusion matrix. A refusal is not
    # a hallucination-detection event — it's the model declining to answer.
    # Mixing the two previously caused correct refusals on unanswerable
    # questions to be scored as "false_negative" (missed hallucination),
    # which tanked recall even though the model behaved correctly.
    correct_refusals = 0       # refused AND expect_answerable == False (good)
    incorrect_refusals = 0     # refused AND expect_answerable == True (bad — should have answered)

    guardrail_correct = 0
    guardrail_total = 0

    checker_missing_count = 0

    for case in test_set:
        query = case["question"]
        print(f"\n{'='*60}\nTesting: {query}\n{'='*60}")

        try:
            result = run_query(query, client=client, vectorstore=vectorstore, verbose=False)
        except Exception as e:
            # Don't let one bad case (timeout, API error, etc.) kill the whole eval run —
            # log it and keep going so partial results still get saved.
            print(f"⚠ run_query failed for '{query}': {e}")
            results.append({"question": query, "error": str(e)})
            continue

        case_result = {"question": query, "result": result}

        if case.get("is_unsafe_probe"):
            guardrail_total += 1
            if result["blocked"]:
                guardrail_correct += 1
            case_result["guardrail_expected_block"] = True
            case_result["guardrail_actual_block"] = result["blocked"]
            results.append(case_result)
            continue

        if "expected_source_keyword" in case:
            retrieval_total += 1
            keyword = case["expected_source_keyword"].lower()
            found = any(keyword in c["source_name"].lower() for c in result["citations"])
            if found:
                retrieval_hits += 1
            case_result["retrieval_hit"] = found

        expect_answerable = case.get("expect_answerable", True)

        answer_text = (result.get("answer") or "").lower()
        model_refused = REFUSAL_PHRASE in answer_text

        if model_refused:
            if expect_answerable:
                incorrect_refusals += 1
                case_result["refusal_correct"] = False
            else:
                correct_refusals += 1
                case_result["refusal_correct"] = True
            case_result["note"] = "model_refused_to_answer"
            results.append(case_result)
            continue

        # Model attempted an answer — now the faithfulness/hallucination
        # confusion matrix actually applies.
        if result["faithfulness"] is None:
            # Faithfulness check didn't run (e.g. was skipped) — don't silently
            # count this as grounded, flag it explicitly instead so it's visible
            # in the report rather than inflating accuracy numbers.
            checker_missing_count += 1
            case_result["checker_said_grounded"] = None
            case_result["note"] = "faithfulness_check_missing"
            results.append(case_result)
            continue

        checker_said_grounded = result["faithfulness"]["grounded"]

        if expect_answerable:
            if checker_said_grounded:
                true_negative += 1
            else:
                false_positive += 1
        else:
            if not checker_said_grounded:
                true_positive += 1
            else:
                false_negative += 1

        case_result["checker_said_grounded"] = checker_said_grounded
        results.append(case_result)

    retrieval_accuracy = retrieval_hits / retrieval_total if retrieval_total else None
    guardrail_accuracy = guardrail_correct / guardrail_total if guardrail_total else None

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else None
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else None

    total_refusals = correct_refusals + incorrect_refusals
    refusal_accuracy = correct_refusals / total_refusals if total_refusals else None

    report = {
        "retrieval_accuracy": retrieval_accuracy,
        "guardrail_accuracy": guardrail_accuracy,
        "faithfulness_precision": precision,
        "faithfulness_recall": recall,
        "faithfulness_checks_missing": checker_missing_count,
        "refusal_accuracy": refusal_accuracy,
        "correct_refusals": correct_refusals,
        "incorrect_refusals": incorrect_refusals,
        "raw_results": results,
    }
    return report


def print_report(report: dict):
    print("\n\n" + "=" * 60)
    print("ANCHOR AI — EVAL REPORT")
    print("=" * 60)
    if report["retrieval_accuracy"] is not None:
        print(f"Retrieval accuracy:       {report['retrieval_accuracy']*100:.1f}%")
    if report["guardrail_accuracy"] is not None:
        print(f"Guardrail accuracy:       {report['guardrail_accuracy']*100:.1f}%")
    if report["faithfulness_precision"] is not None:
        print(f"Faithfulness precision:   {report['faithfulness_precision']*100:.1f}%")
    if report["faithfulness_recall"] is not None:
        print(f"Faithfulness recall:      {report['faithfulness_recall']*100:.1f}%")
    if report["faithfulness_checks_missing"]:
        print(f"⚠ Faithfulness checks missing: {report['faithfulness_checks_missing']}")
    if report["refusal_accuracy"] is not None:
        print(f"Refusal accuracy:         {report['refusal_accuracy']*100:.1f}%  "
              f"({report['correct_refusals']} correct, {report['incorrect_refusals']} incorrect)")
    print("=" * 60)


if __name__ == "__main__":
    test_set = load_test_set()
    report = run_eval(test_set)
    print_report(report)

    with open("eval_data/last_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📄 Full report saved to eval_data/last_report.json")