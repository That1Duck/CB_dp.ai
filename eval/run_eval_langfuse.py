import json
from pathlib import Path

from src.interface.run_query_langfuse import run_query_with_trace
from langfuse import observe, get_client
from src.utils.logging import configure_logging, get_logger

configure_logging("INFO", "logs/eval.log")
logger = get_logger("eval")

DATASET = Path("eval/dataset.json")

def norm_url(url):
    return url.strip().rstrip("/").lower()

def rank_in_candidates(cands, expected_url):
    if not expected_url or not cands:
        return None
    exp_url = norm_url(expected_url)
    for id, url in enumerate(cands, 1):
        if norm_url(url.get("url")) == exp_url:
            return id
    return None

@observe
def run_case(case):
    logger.info("Start of case", extra={"id": case["id"]})
    client = get_client()
    client.update_current_trace(
        name = "eval_case",
        metadata={
            "case_id": case.get("id"),
            "dataset": "v1",
            "expected_intent": case.get("expected_intent"),
            "expected_url": case.get("expected_url")
        },
        tags=["eval", "deeplearning.ai"]
    )

    case_id = case.get("id")

    result = run_query_with_trace(case["query"], meta = {"case_id":case.get("id"), "dataset": "v1"})

    expected_intent = case.get("expected_intent")
    predicted_status = result.get("status")
    predicted_intent = (result.get("decision") or {}).get("intent") or result.get("intent")
    expected_url = case.get("expected_url")

    metrics = client.start_span(name="Metrics")
    try:
        # Intent Accuracy
        if expected_intent in ("course", "blog"):
            acc = int(predicted_intent == expected_intent)
            client.score_current_trace(name="Intent_Accuracy", value=acc,
                                       comment=f"case_id: {case_id}")


        # OOS Refusal Accuracy
        if expected_intent == "oos":
            oos = int(predicted_intent == expected_intent)
            client.score_current_trace(name="OOS_Refusal", value=oos,
                                       comment=f"case_id: {case_id}")

        # Retriever
        r = rank_in_candidates(result.get("candidates"), expected_url)
        hit3 = int(r is not None and r <= 3) if expected_url else None
        if hit3 is not None:
            client.score_current_trace(name="hit@3", value=hit3,
                                       comment=f"case_id: {case_id}")

        # Final Answer Accuracy
        url_match = int(norm_url(result.get("url") or "") == norm_url(expected_url)) if expected_url else None
        if expected_url:
            client.score_current_trace(
                name="URL match",
                value=url_match,
                comment=f"case_id: {case_id}"
            )

        metrics.update(output={
            "Intent_Accuracy": acc if expected_intent in ("course", "blog") else None,
            "OOS_Refusal": oos if expected_intent == "oos" else None,
            "hit@3": hit3,
            "URL_match": url_match,
        })
    finally:
        metrics.end()

    client.update_current_span(output={
        "result_summary":{
            "status":result.get("status"),
            "intent":(result.get("decision")or {}).get("intent") or result.get("intent"),
            "title": result.get("title"),
            "url":result.get("url")
        }
    })
    logger.info("Case completed", extra={
        "id": case["id"],
        "status": result.get("status"),
        "intent": result.get("intent"),
        "url": result.get("url"),
    })

@observe()
def main():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    for case in cases:
        run_case(case)

    get_client().flush()

if __name__ == "__main__":
    main()