from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent import (
    AgentEvaluation,
    EVAL_AGENT_A_MODEL,
    EVAL_AGENT_B_MODEL,
    PeerReview,
    ask_agent,
    evaluate_with_agent_a,
    evaluate_with_agent_b,
    review_agent_a_work,
    review_agent_b_work,
)


DATASET_PATH = Path("data/eval_set.jsonl")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "eval_results.json"
RESEARCH_REPORT_PATH = RESULTS_DIR / "research_report.md"
CRITERIA = [
    "factual_correctness",
    "completeness",
    "specificity",
    "format_compliance",
    "unsupported_content",
]


@dataclass
class EvalItem:
    id: int
    category: str
    question: str
    expected_answer: str
    chatbot_answer: str


@dataclass
class EvalResult:
    id: int
    category: str
    question: str
    expected_answer: str
    chatbot_answer: str
    agent_a_evaluation: AgentEvaluation
    agent_b_evaluation: AgentEvaluation
    agent_a_review_of_b: PeerReview
    agent_b_review_of_a: PeerReview
    consensus_passed: bool
    needs_manual_review: bool
    evaluator_disagreement: bool
    peer_review_disagreement: bool
    confidence_gap: float


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            question = row["question"]
            chatbot_answer = row.get("chatbot_answer")

            if chatbot_answer is None:
                chatbot_answer = ask_agent(question)

            items.append(
                EvalItem(
                    id=int(row["id"]),
                    category=row.get("category", "algemeen"),
                    question=question,
                    expected_answer=row["expected_answer"],
                    chatbot_answer=chatbot_answer,
                )
            )

    return items


def evaluate_item(item: EvalItem) -> EvalResult:
    agent_a_evaluation = evaluate_with_agent_a(
        item.question,
        item.expected_answer,
        item.chatbot_answer,
    )
    agent_b_evaluation = evaluate_with_agent_b(
        item.question,
        item.expected_answer,
        item.chatbot_answer,
    )

    agent_a_review_of_b = review_agent_b_work(
        item.question,
        item.expected_answer,
        item.chatbot_answer,
        agent_b_evaluation,
    )
    agent_b_review_of_a = review_agent_a_work(
        item.question,
        item.expected_answer,
        item.chatbot_answer,
        agent_a_evaluation,
    )

    return build_eval_result(
        item,
        agent_a_evaluation,
        agent_b_evaluation,
        agent_a_review_of_b,
        agent_b_review_of_a,
    )


def _llm_error_evaluation(evaluator: str, model: str, exc: BaseException) -> AgentEvaluation:
    return AgentEvaluation(
        evaluator=evaluator,
        model=model,
        passed=False,
        verdict="fail",
        confidence_score=0.0,
        reasoning=f"LLM judge call failed or returned invalid JSON: {exc}",
        issues=["llm_error"],
        criterion_scores={criterion: 0.0 for criterion in CRITERIA},
        missing_facts=[],
        unsupported_claims=[],
        format_issues=["judge did not return parseable JSON"],
    )


def _llm_error_review(
    reviewer: str,
    model: str,
    reviewed_evaluator: str,
    exc: BaseException,
) -> PeerReview:
    return PeerReview(
        reviewer=reviewer,
        model=model,
        reviewed_evaluator=reviewed_evaluator,
        agrees=False,
        confidence_score=0.0,
        corrected_verdict="unchanged",
        reasoning=f"LLM peer-review call failed or returned invalid JSON: {exc}",
        issues=["llm_error"],
        decisive_errors=["reviewer did not return parseable JSON"],
    )


async def evaluate_item_async(item: EvalItem) -> EvalResult:
    agent_a_task = asyncio.to_thread(
        evaluate_with_agent_a,
        item.question,
        item.expected_answer,
        item.chatbot_answer,
    )
    agent_b_task = asyncio.to_thread(
        evaluate_with_agent_b,
        item.question,
        item.expected_answer,
        item.chatbot_answer,
    )
    agent_a_result, agent_b_result = await asyncio.gather(
        agent_a_task,
        agent_b_task,
        return_exceptions=True,
    )
    agent_a_evaluation = (
        _llm_error_evaluation("agent_a_strict_llm", EVAL_AGENT_A_MODEL, agent_a_result)
        if isinstance(agent_a_result, BaseException)
        else agent_a_result
    )
    agent_b_evaluation = (
        _llm_error_evaluation("agent_b_semantic_llm", EVAL_AGENT_B_MODEL, agent_b_result)
        if isinstance(agent_b_result, BaseException)
        else agent_b_result
    )

    agent_a_review_task = asyncio.to_thread(
        review_agent_b_work,
        item.question,
        item.expected_answer,
        item.chatbot_answer,
        agent_b_evaluation,
    )
    agent_b_review_task = asyncio.to_thread(
        review_agent_a_work,
        item.question,
        item.expected_answer,
        item.chatbot_answer,
        agent_a_evaluation,
    )
    agent_a_review_result, agent_b_review_result = await asyncio.gather(
        agent_a_review_task,
        agent_b_review_task,
        return_exceptions=True,
    )
    agent_a_review_of_b = (
        _llm_error_review(
            "agent_a_strict_llm",
            EVAL_AGENT_A_MODEL,
            agent_b_evaluation.evaluator,
            agent_a_review_result,
        )
        if isinstance(agent_a_review_result, BaseException)
        else agent_a_review_result
    )
    agent_b_review_of_a = (
        _llm_error_review(
            "agent_b_semantic_llm",
            EVAL_AGENT_B_MODEL,
            agent_a_evaluation.evaluator,
            agent_b_review_result,
        )
        if isinstance(agent_b_review_result, BaseException)
        else agent_b_review_result
    )

    return build_eval_result(
        item,
        agent_a_evaluation,
        agent_b_evaluation,
        agent_a_review_of_b,
        agent_b_review_of_a,
    )


async def evaluate_dataset_async(
    dataset: list[EvalItem],
    parallel_items: int,
) -> list[EvalResult]:
    semaphore = asyncio.Semaphore(max(parallel_items, 1))

    async def evaluate_with_limit(index: int, item: EvalItem) -> tuple[int, EvalResult]:
        async with semaphore:
            return index, await evaluate_item_async(item)

    indexed_results = await asyncio.gather(
        *(evaluate_with_limit(index, item) for index, item in enumerate(dataset))
    )
    return [result for _, result in sorted(indexed_results, key=lambda pair: pair[0])]


def build_eval_result(
    item: EvalItem,
    agent_a_evaluation: AgentEvaluation,
    agent_b_evaluation: AgentEvaluation,
    agent_a_review_of_b: PeerReview,
    agent_b_review_of_a: PeerReview,
) -> EvalResult:
    evaluators_agree = agent_a_evaluation.passed == agent_b_evaluation.passed
    reviews_agree = agent_a_review_of_b.agrees and agent_b_review_of_a.agrees
    consensus_passed = agent_a_evaluation.passed and agent_b_evaluation.passed and reviews_agree
    evaluator_disagreement = not evaluators_agree
    peer_review_disagreement = not reviews_agree
    needs_manual_review = evaluator_disagreement or peer_review_disagreement
    confidence_gap = abs(
        agent_a_evaluation.confidence_score - agent_b_evaluation.confidence_score
    )

    return EvalResult(
        id=item.id,
        category=item.category,
        question=item.question,
        expected_answer=item.expected_answer,
        chatbot_answer=item.chatbot_answer,
        agent_a_evaluation=agent_a_evaluation,
        agent_b_evaluation=agent_b_evaluation,
        agent_a_review_of_b=agent_a_review_of_b,
        agent_b_review_of_a=agent_b_review_of_a,
        consensus_passed=consensus_passed,
        needs_manual_review=needs_manual_review,
        evaluator_disagreement=evaluator_disagreement,
        peer_review_disagreement=peer_review_disagreement,
        confidence_gap=round(confidence_gap, 4),
    )


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _bias_direction(agent_a_value: float, agent_b_value: float) -> str:
    if agent_a_value > agent_b_value:
        return "agent_a"
    if agent_b_value > agent_a_value:
        return "agent_b"
    return "none"


def _risk_level(gap: float) -> str:
    if gap >= 0.20:
        return "high"
    if gap >= 0.10:
        return "medium"
    if gap >= 0.05:
        return "low"
    return "none"


def calculate_disagreement_metrics(results: list[EvalResult]) -> dict:
    total = len(results)
    evaluator_disagreements = sum(1 for r in results if r.evaluator_disagreement)
    peer_review_disagreements = sum(1 for r in results if r.peer_review_disagreement)
    manual_review = sum(1 for r in results if r.needs_manual_review)

    return {
        "evaluator_disagreements": evaluator_disagreements,
        "evaluator_disagreement_rate": _rate(evaluator_disagreements, total),
        "peer_review_disagreements": peer_review_disagreements,
        "peer_review_disagreement_rate": _rate(peer_review_disagreements, total),
        "needs_manual_review": manual_review,
        "manual_review_rate": _rate(manual_review, total),
        "average_confidence_gap": _average([r.confidence_gap for r in results]),
    }


def _kappa_interpretation(kappa: float) -> str:
    if kappa < 0.0:
        return "poor"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost_perfect"


def calculate_inter_rater_reliability(results: list[EvalResult]) -> dict:
    total = len(results)
    both_pass = sum(
        1
        for r in results
        if r.agent_a_evaluation.passed and r.agent_b_evaluation.passed
    )
    both_fail = sum(
        1
        for r in results
        if not r.agent_a_evaluation.passed and not r.agent_b_evaluation.passed
    )
    agent_a_only_pass = sum(
        1
        for r in results
        if r.agent_a_evaluation.passed and not r.agent_b_evaluation.passed
    )
    agent_b_only_pass = sum(
        1
        for r in results
        if not r.agent_a_evaluation.passed and r.agent_b_evaluation.passed
    )

    if not total:
        return {
            "metric": "cohen_kappa",
            "cohen_kappa": 0.0,
            "interpretation": "not_available",
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "positive_agreement": 0.0,
            "negative_agreement": 0.0,
            "confusion_matrix": {
                "both_pass": 0,
                "both_fail": 0,
                "agent_a_pass_agent_b_fail": 0,
                "agent_a_fail_agent_b_pass": 0,
            },
        }

    observed_agreement = (both_pass + both_fail) / total
    agent_a_pass_rate = (both_pass + agent_a_only_pass) / total
    agent_a_fail_rate = 1 - agent_a_pass_rate
    agent_b_pass_rate = (both_pass + agent_b_only_pass) / total
    agent_b_fail_rate = 1 - agent_b_pass_rate
    expected_agreement = (
        agent_a_pass_rate * agent_b_pass_rate
        + agent_a_fail_rate * agent_b_fail_rate
    )
    if expected_agreement == 1.0:
        cohen_kappa = 1.0 if observed_agreement == 1.0 else 0.0
    else:
        cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    positive_denominator = (2 * both_pass) + agent_a_only_pass + agent_b_only_pass
    negative_denominator = (2 * both_fail) + agent_a_only_pass + agent_b_only_pass

    return {
        "metric": "cohen_kappa",
        "cohen_kappa": round(cohen_kappa, 4),
        "interpretation": _kappa_interpretation(cohen_kappa),
        "observed_agreement": round(observed_agreement, 4),
        "expected_agreement": round(expected_agreement, 4),
        "positive_agreement": round((2 * both_pass) / positive_denominator, 4)
        if positive_denominator
        else 0.0,
        "negative_agreement": round((2 * both_fail) / negative_denominator, 4)
        if negative_denominator
        else 0.0,
        "confusion_matrix": {
            "both_pass": both_pass,
            "both_fail": both_fail,
            "agent_a_pass_agent_b_fail": agent_a_only_pass,
            "agent_a_fail_agent_b_pass": agent_b_only_pass,
        },
    }


def calculate_bias_metrics(results: list[EvalResult]) -> dict:
    total = len(results)
    agent_a_pass_rate = _rate(sum(1 for r in results if r.agent_a_evaluation.passed), total)
    agent_b_pass_rate = _rate(sum(1 for r in results if r.agent_b_evaluation.passed), total)
    pass_rate_gap = round(abs(agent_a_pass_rate - agent_b_pass_rate), 4)

    agent_a_confidence = _average(
        [r.agent_a_evaluation.confidence_score for r in results]
    )
    agent_b_confidence = _average(
        [r.agent_b_evaluation.confidence_score for r in results]
    )
    confidence_gap = round(abs(agent_a_confidence - agent_b_confidence), 4)

    agent_a_review_agree_rate = _rate(
        sum(1 for r in results if r.agent_a_review_of_b.agrees), total
    )
    agent_b_review_agree_rate = _rate(
        sum(1 for r in results if r.agent_b_review_of_a.agrees), total
    )
    review_agreement_gap = round(
        abs(agent_a_review_agree_rate - agent_b_review_agree_rate), 4
    )

    by_category = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    category_bias = {}
    category_pass_rates = []
    category_disagreement_rates = []
    for category, category_results in by_category.items():
        cat_total = len(category_results)
        cat_agent_a_pass_rate = _rate(
            sum(1 for r in category_results if r.agent_a_evaluation.passed), cat_total
        )
        cat_agent_b_pass_rate = _rate(
            sum(1 for r in category_results if r.agent_b_evaluation.passed), cat_total
        )
        cat_consensus_pass_rate = _rate(
            sum(1 for r in category_results if r.consensus_passed), cat_total
        )
        cat_disagreement_rate = _rate(
            sum(1 for r in category_results if r.evaluator_disagreement), cat_total
        )
        cat_manual_review_rate = _rate(
            sum(1 for r in category_results if r.needs_manual_review), cat_total
        )
        cat_pass_rate_gap = round(
            abs(cat_agent_a_pass_rate - cat_agent_b_pass_rate), 4
        )

        category_pass_rates.append(cat_consensus_pass_rate)
        category_disagreement_rates.append(cat_disagreement_rate)
        category_bias[category] = {
            "total": cat_total,
            "agent_a_pass_rate": cat_agent_a_pass_rate,
            "agent_b_pass_rate": cat_agent_b_pass_rate,
            "agent_pass_rate_gap": cat_pass_rate_gap,
            "consensus_pass_rate": cat_consensus_pass_rate,
            "evaluator_disagreement_rate": cat_disagreement_rate,
            "manual_review_rate": cat_manual_review_rate,
        }

    category_pass_rate_range = round(
        max(category_pass_rates) - min(category_pass_rates), 4
    ) if category_pass_rates else 0.0
    category_disagreement_rate_range = round(
        max(category_disagreement_rates) - min(category_disagreement_rates), 4
    ) if category_disagreement_rates else 0.0

    risk_flags = []
    if pass_rate_gap >= 0.05:
        risk_flags.append(
            f"leniency gap: {_bias_direction(agent_a_pass_rate, agent_b_pass_rate)} passes more often"
        )
    if confidence_gap >= 0.10:
        risk_flags.append(
            f"confidence gap: {_bias_direction(agent_a_confidence, agent_b_confidence)} is more confident"
        )
    if review_agreement_gap >= 0.10:
        risk_flags.append(
            "peer-review agreement differs between judges"
        )
    if category_pass_rate_range >= 0.20:
        risk_flags.append("large category pass-rate range")
    if category_disagreement_rate_range >= 0.20:
        risk_flags.append("large category disagreement-rate range")

    return {
        "agent_a_pass_rate": agent_a_pass_rate,
        "agent_b_pass_rate": agent_b_pass_rate,
        "pass_rate_gap": pass_rate_gap,
        "leniency_bias_toward": _bias_direction(agent_a_pass_rate, agent_b_pass_rate),
        "leniency_bias_risk": _risk_level(pass_rate_gap),
        "agent_a_average_confidence": agent_a_confidence,
        "agent_b_average_confidence": agent_b_confidence,
        "confidence_gap": confidence_gap,
        "confidence_bias_toward": _bias_direction(agent_a_confidence, agent_b_confidence),
        "confidence_bias_risk": _risk_level(confidence_gap),
        "agent_a_review_agree_rate": agent_a_review_agree_rate,
        "agent_b_review_agree_rate": agent_b_review_agree_rate,
        "review_agreement_gap": review_agreement_gap,
        "review_bias_risk": _risk_level(review_agreement_gap),
        "category_pass_rate_range": category_pass_rate_range,
        "category_disagreement_rate_range": category_disagreement_rate_range,
        "category_bias": category_bias,
        "risk_flags": risk_flags,
    }


def summarize_results(results: list[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.consensus_passed)
    failed = total - passed
    needs_manual_review = sum(1 for r in results if r.needs_manual_review)
    accuracy = passed / total if total else 0

    by_category = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    category_summary = {}
    for category, category_results in by_category.items():
        cat_total = len(category_results)
        cat_passed = sum(1 for r in category_results if r.consensus_passed)
        cat_manual_review = sum(1 for r in category_results if r.needs_manual_review)
        cat_evaluator_disagreement = sum(
            1 for r in category_results if r.evaluator_disagreement
        )
        cat_peer_review_disagreement = sum(
            1 for r in category_results if r.peer_review_disagreement
        )
        cat_accuracy = cat_passed / cat_total if cat_total else 0

        category_summary[category] = {
            "total": cat_total,
            "passed": cat_passed,
            "failed": cat_total - cat_passed,
            "needs_manual_review": cat_manual_review,
            "manual_review_rate": _rate(cat_manual_review, cat_total),
            "evaluator_disagreement_rate": _rate(
                cat_evaluator_disagreement,
                cat_total,
            ),
            "peer_review_disagreement_rate": _rate(
                cat_peer_review_disagreement,
                cat_total,
            ),
            "accuracy": round(cat_accuracy, 4),
        }

    weakest_categories = sorted(
        category_summary.items(),
        key=lambda item: item[1]["accuracy"],
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "needs_manual_review": needs_manual_review,
        "accuracy": round(accuracy, 4),
        "disagreement_metrics": calculate_disagreement_metrics(results),
        "inter_rater_reliability": calculate_inter_rater_reliability(results),
        "bias_metrics": calculate_bias_metrics(results),
        "category_summary": category_summary,
        "weakest_categories": weakest_categories[:3],
    }


def generate_improvement_advice(summary: dict, results: list[EvalResult]) -> list[str]:
    advice = []
    accuracy = summary["accuracy"]

    if accuracy >= 0.90:
        advice.append("Score is strong. Add stricter edge cases and more question variation.")
    elif accuracy >= 0.75:
        advice.append("Score is reasonable. Improve the weakest categories first.")
    else:
        advice.append("Score is low. Check prompt, context, source data, and answer format rules first.")

    if summary["needs_manual_review"]:
        advice.append(
            f"Manual review is needed for {summary['needs_manual_review']} item(s) where evaluators or peer reviews disagreed."
        )

    disagreement = summary["disagreement_metrics"]
    if disagreement["evaluator_disagreement_rate"] >= 0.05:
        advice.append(
            "Evaluator disagreement is above 5%. Inspect those items before treating the score as stable."
        )
    if disagreement["peer_review_disagreement_rate"] >= 0.05:
        advice.append(
            "Peer-review disagreement is above 5%. The judges are not consistently validating each other's decisions."
        )

    reliability = summary["inter_rater_reliability"]
    if reliability["cohen_kappa"] < 0.61:
        advice.append(
            "Inter-rater reliability is below substantial agreement. Inspect judge prompts, ambiguous gold answers, and disagreement cases before trusting the consensus score."
        )

    bias = summary["bias_metrics"]
    if bias["risk_flags"]:
        advice.append("Bias risk flags: " + "; ".join(bias["risk_flags"]) + ".")

    weakest_categories = [
        (category, stats)
        for category, stats in summary["weakest_categories"]
        if stats["accuracy"] < 1.0
    ]

    for category, stats in weakest_categories:
        advice.append(
            f"Category '{category}' scores {stats['accuracy'] * 100:.1f}%. "
            "Review failed chatbot answers and add focused test questions."
        )

    failed_results = [r for r in results if not r.consensus_passed][:5]

    if failed_results:
        advice.append("First failed or non-consensus items:")
        for result in failed_results:
            advice.append(
                f"- Question {result.id}: expected '{result.expected_answer}', chatbot answered '{result.chatbot_answer}'."
            )

    return advice


def request_user_approval(summary: dict) -> dict:
    print("\n=== APPROVAL REQUIRED ===")
    print(
        "Both evaluator agents have evaluated the chatbot answers and reviewed each other's work."
    )
    print(
        f"Consensus score: {summary['passed']}/{summary['total']} "
        f"({summary['accuracy'] * 100:.1f}%)."
    )

    try:
        answer = input("Approve these evaluation results? [y/N]: ").strip().lower()
    except EOFError:
        return {
            "status": "pending",
            "approved": False,
            "note": "No interactive approval input was available.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    approved = answer in {"y", "yes", "j", "ja"}

    return {
        "status": "approved" if approved else "rejected",
        "approved": approved,
        "note": "Approved by user." if approved else "Rejected or left unapproved by user.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_results(
    results: list[EvalResult],
    summary: dict,
    advice: list[str],
    approval: dict,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    payload = {
        "summary": summary,
        "advice": advice,
        "approval": approval,
        "results": [asdict(result) for result in results],
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    RESEARCH_REPORT_PATH.write_text(
        build_research_report(summary, advice, approval),
        encoding="utf-8",
    )


def build_research_report(summary: dict, advice: list[str], approval: dict) -> str:
    disagreement = summary["disagreement_metrics"]
    reliability = summary["inter_rater_reliability"]
    bias = summary["bias_metrics"]
    generated_at = datetime.now(timezone.utc).isoformat()

    category_lines = []
    for category, stats in summary["category_summary"].items():
        category_lines.append(
            f"- {category}: pass_rate={stats['accuracy']:.4f}, "
            f"manual_review_rate={stats['manual_review_rate']:.4f}, "
            f"evaluator_disagreement_rate={stats['evaluator_disagreement_rate']:.4f}"
        )

    risk_flags = bias["risk_flags"] or ["none"]

    return f"""# Research Evaluation Report

Generated: {generated_at}

## Hypothesis

Using two independent LLM judges plus reciprocal peer review gives a more reliable evaluation of chatbot answers than a single judge, because disagreement and bias metrics make unstable judgments visible before approval.

## Methods

- Dataset: {summary['total']} question-answer items from `data/eval_set.jsonl`.
- Judge design: Agent A is strict on factual and format compliance. Agent B focuses on semantic correctness.
- Decision rule: an item passes only when both judges pass the answer and both peer reviews agree.
- Disagreement metrics: evaluator disagreement rate, peer-review disagreement rate, manual-review rate, and average confidence gap.
- Inter-rater reliability: Cohen's kappa for Agent A vs Agent B pass/fail decisions, with observed and chance-expected agreement.
- Bias metrics: pass-rate gap, leniency direction, confidence gap, peer-review agreement gap, category pass-rate range, and category disagreement-rate range.
- Reproducibility: full per-item judgments, reasoning, issues, metrics, and approval status are stored in `results/eval_results.json`.

## Results

- Consensus score: {summary['passed']}/{summary['total']} ({summary['accuracy'] * 100:.1f}%).
- Evaluator disagreement rate: {disagreement['evaluator_disagreement_rate'] * 100:.1f}% ({disagreement['evaluator_disagreements']} items).
- Inter-rater reliability: Cohen's kappa={reliability['cohen_kappa']:.4f} ({reliability['interpretation']}), observed_agreement={reliability['observed_agreement'] * 100:.1f}%, expected_agreement={reliability['expected_agreement'] * 100:.1f}%.
- Peer-review disagreement rate: {disagreement['peer_review_disagreement_rate'] * 100:.1f}% ({disagreement['peer_review_disagreements']} items).
- Manual-review rate: {disagreement['manual_review_rate'] * 100:.1f}% ({disagreement['needs_manual_review']} items).
- Average confidence gap: {disagreement['average_confidence_gap']:.4f}.
- Agent A pass rate: {bias['agent_a_pass_rate'] * 100:.1f}%.
- Agent B pass rate: {bias['agent_b_pass_rate'] * 100:.1f}%.
- Leniency pass-rate gap: {bias['pass_rate_gap'] * 100:.1f}% toward {bias['leniency_bias_toward']}.
- Confidence gap: {bias['confidence_gap']:.4f} toward {bias['confidence_bias_toward']}.
- Review agreement gap: {bias['review_agreement_gap'] * 100:.1f}%.
- Category pass-rate range: {bias['category_pass_rate_range'] * 100:.1f}%.
- Category disagreement-rate range: {bias['category_disagreement_rate_range'] * 100:.1f}%.
- Bias risk flags: {'; '.join(risk_flags)}.
- Approval status: {approval['status']}.

## Category Results

{chr(10).join(category_lines)}

## Improvement Advice

{chr(10).join(f'- {item}' for item in advice)}

## Limitations

- LLM judges are probabilistic and may share training-data or prompt-induced biases, especially when both agents use the same base model.
- Ollama local model quality depends on the model pulled, quantization, hardware, and runtime settings.
- The dataset size and category balance limit statistical confidence; larger stratified datasets are needed for stronger claims.
- The gold answer is treated as authoritative. If the gold answer is ambiguous or wrong, the judges may penalize a valid chatbot answer.
- Disagreement and bias metrics diagnose evaluation reliability; they do not prove real-world user safety by themselves.
"""


def print_report(summary: dict, advice: list[str]) -> None:
    disagreement = summary["disagreement_metrics"]
    reliability = summary["inter_rater_reliability"]
    bias = summary["bias_metrics"]

    print("\n=== EVAL REPORT ===")
    print(f"Total:               {summary['total']}")
    print(f"Consensus passed:    {summary['passed']}")
    print(f"Failed/non-consensus:{summary['failed']}")
    print(f"Manual review:       {summary['needs_manual_review']}")
    print(f"Score:               {summary['accuracy'] * 100:.1f}%")
    print(f"Evaluator disagrees: {disagreement['evaluator_disagreement_rate'] * 100:.1f}%")
    print(
        "Inter-rater kappa:   "
        f"{reliability['cohen_kappa']:.4f} ({reliability['interpretation']})"
    )
    print(f"Observed agreement:  {reliability['observed_agreement'] * 100:.1f}%")
    print(f"Peer review disagrees:{disagreement['peer_review_disagreement_rate'] * 100:.1f}%")
    print(f"Leniency gap:        {bias['pass_rate_gap'] * 100:.1f}% ({bias['leniency_bias_toward']})")
    print(f"Confidence gap:      {bias['confidence_gap']:.4f} ({bias['confidence_bias_toward']})")

    print("\n=== SCORE PER CATEGORY ===")
    for category, stats in summary["category_summary"].items():
        print(
            f"- {category}: {stats['passed']}/{stats['total']} "
            f"({stats['accuracy'] * 100:.1f}%), "
            f"manual review: {stats['needs_manual_review']}, "
            f"judge disagreement: {stats['evaluator_disagreement_rate'] * 100:.1f}%"
        )

    print("\n=== IMPROVEMENT ADVICE ===")
    for item in advice:
        print(item)


def run_eval_loop() -> None:
    dataset = load_dataset(DATASET_PATH)
    max_items = int(os.getenv("EVAL_MAX_ITEMS", "1") or "1")
    if max_items > 0:
        dataset = dataset[:max_items]

    if len(dataset) != 1:
        print(f"Warning: dataset contains {len(dataset)} questions, not exactly 1.")

    parallel_items = int(os.getenv("EVAL_PARALLEL_ITEMS", "4") or "4")
    if parallel_items < 1:
        parallel_items = 1
    print(f"Running async eval with up to {parallel_items} item(s) in parallel.")
    results = asyncio.run(evaluate_dataset_async(dataset, parallel_items))
    summary = summarize_results(results)
    advice = generate_improvement_advice(summary, results)

    print_report(summary, advice)
    if os.getenv("EVAL_SKIP_APPROVAL", "").strip().lower() in {"1", "true", "yes"}:
        approval = {
            "status": "skipped",
            "approved": False,
            "note": "Approval skipped because EVAL_SKIP_APPROVAL is set.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        approval = request_user_approval(summary)
    save_results(results, summary, advice, approval)

    print(f"\nApproval status: {approval['status']}")
    print(f"Results saved in: {RESULTS_PATH}")
    print(f"Research report saved in: {RESEARCH_REPORT_PATH}")


if __name__ == "__main__":
    run_eval_loop()
