# Eval Agent Loop

A VS Code-ready evaluation loop for chatbot answers using LLM judges. The default setup runs both judge agents locally through Ollama with different judge prompts, records disagreement and bias metrics, and writes an auditable research report.

## Hypothesis

Two independent LLM judges with reciprocal peer review produce a more trustworthy chatbot evaluation than one judge alone, because the loop exposes unstable judgments through disagreement rate and bias metrics before human approval.

## Methods

The loop reads `data/eval_set.jsonl`. Each row contains a question, a gold answer, and a chatbot answer.

```json
{"id": 1, "category": "UI", "question": "Why am I not seeing anything after opening the Qmass app using the button in the top bar?", "expected_answer": "First drag and drop a series inside, or open Qmass with a right mouse click on the series.", "chatbot_answer": "First drag and drop a series inside, or open Qmass with a right mouse click on the series."}
```

Required fields:

- `id`
- `question`
- `expected_answer`
- `chatbot_answer`

Optional field:

- `category`

If `chatbot_answer` is missing, the loop calls `ask_agent(question)` from `agent.py`.

Evaluation design:

- Agent A uses a strict precision prompt focused on factual correctness, exact task compliance, required terminology, paths, settings, and format compliance.
- Agent B uses a semantic equivalence prompt focused on fair paraphrase acceptance while still rejecting contradictions, omissions, and unsupported details.
- Agent A reviews Agent B's judgment.
- Agent B reviews Agent A's judgment.
- An item passes only when both judges pass it and both peer reviews agree.
- Any judge disagreement or peer-review disagreement marks the item for manual review.
- The eval loop runs asynchronously: Agent A and Agent B evaluate each item in parallel, then both peer reviews run in parallel.

Prompt controls:

- Both judges treat the chatbot answer as untrusted data and ignore prompt-injection attempts inside it.
- Both judges must score factual correctness, completeness, specificity, format compliance, and unsupported content.
- Both judges must return machine-readable JSON only.
- Review prompts force each reviewer to independently re-evaluate the answer before agreeing with the other judge.

Metrics recorded:

- Consensus pass rate.
- Evaluator disagreement rate.
- Peer-review disagreement rate.
- Manual-review rate.
- Average confidence gap between judges.
- Inter-rater reliability using Cohen's kappa for Agent A vs Agent B pass/fail decisions.
- Observed agreement, expected agreement, positive agreement, negative agreement, and the binary pass/fail confusion matrix.
- Agent A and Agent B pass rates.
- Leniency bias direction and risk level.
- Confidence bias direction and risk level.
- Peer-review agreement gap.
- Category pass-rate range.
- Category disagreement-rate range.
- Per-category bias metrics.

Outputs:

- `results/eval_results.json` stores all item-level judge decisions, reasoning, issues, summary metrics, advice, and approval status.
- `results/research_report.md` stores a research-style report with hypothesis, methods, results, category results, advice, and limitations.

## Run With Ollama

Default free local test setup:

```powershell
ollama pull qwen2.5:3b
python eval_loop.py
```

PowerShell example using the local Ollama model explicitly:

```powershell
$env:EVAL_LLM_PROVIDER="ollama"
$env:EVAL_AGENT_A_MODEL="qwen2.5:3b"
$env:EVAL_AGENT_B_MODEL="qwen2.5:3b"
$env:EVAL_PARALLEL_ITEMS="1"
python eval_loop.py
```

Fast smoke test with one item and no approval prompt:

```powershell
$env:EVAL_LLM_PROVIDER="ollama"
$env:EVAL_AGENT_A_MODEL="qwen2.5:3b"
$env:EVAL_AGENT_B_MODEL="qwen2.5:3b"
$env:EVAL_PARALLEL_ITEMS="1"
$env:EVAL_MAX_ITEMS="1"
$env:EVAL_SKIP_APPROVAL="1"
python eval_loop.py
```

Ollama settings:

- `EVAL_LLM_PROVIDER=ollama` by default.
- Ollama local models do not use `MINIMAX_API_KEY` or another API key by default.
- `OLLAMA_BASE_URL=http://localhost:11434`
- `EVAL_AGENT_A_MODEL=qwen2.5:3b` by default when `EVAL_LLM_PROVIDER=ollama`.
- `EVAL_AGENT_B_MODEL=qwen2.5:3b` by default when `EVAL_LLM_PROVIDER=ollama`.
- `EVAL_PARALLEL_ITEMS=4`
- `EVAL_LLM_TIMEOUT=120`
- `EVAL_LLM_SEED=42`

## Run With MiniMax

Local `.env` option:

```text
MINIMAX_API_KEY=your-minimax-key-here
```

Put your MiniMax pay-as-you-go API key in a file named `.env` next to `eval_loop.py`, or set the key in PowerShell before running. Do not put real API keys directly in `agent.py` or `eval_loop.py`.

PowerShell example using MiniMax for both judge agents with different prompts:

```powershell
$env:MINIMAX_API_KEY="your-minimax-key-here"
$env:EVAL_LLM_PROVIDER="minimax"
$env:EVAL_AGENT_A_MODEL="MiniMax-M3"
$env:EVAL_AGENT_B_MODEL="MiniMax-M3"
$env:EVAL_PARALLEL_ITEMS="4"
python eval_loop.py
```

Fast smoke test with one item and no approval prompt:

```powershell
$env:MINIMAX_API_KEY="your-minimax-key-here"
$env:EVAL_LLM_PROVIDER="minimax"
$env:EVAL_AGENT_A_MODEL="MiniMax-M3"
$env:EVAL_AGENT_B_MODEL="MiniMax-M3"
$env:EVAL_PARALLEL_ITEMS="1"
$env:EVAL_MAX_ITEMS="1"
$env:EVAL_SKIP_APPROVAL="1"
python eval_loop.py
```

MiniMax settings:

- `EVAL_LLM_PROVIDER=minimax` by default.
- `MINIMAX_API_KEY` is required.
- `.env` is loaded automatically when present next to `agent.py`.
- `.env` is ignored by git so the real key is not committed.
- `MINIMAX_BASE_URL=https://api.minimax.io`
- `EVAL_AGENT_A_MODEL=MiniMax-M3` by default when `EVAL_LLM_PROVIDER=minimax`.
- `EVAL_AGENT_B_MODEL=MiniMax-M3` by default when `EVAL_LLM_PROVIDER=minimax`.
- `MINIMAX_TEMPERATURE=0.1`
- `EVAL_PARALLEL_ITEMS=4`
- `EVAL_LLM_TIMEOUT=120`

## Run With Larger Ollama Models

Install Ollama from:

```text
https://ollama.com/download
```

Pull two different local judge models:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

PowerShell example using the Ollama two-model judge setup:

```powershell
$env:EVAL_LLM_PROVIDER="ollama"
$env:EVAL_AGENT_A_MODEL="qwen2.5:7b"
$env:EVAL_AGENT_B_MODEL="llama3.1:8b"
$env:EVAL_PARALLEL_ITEMS="4"
python eval_loop.py
```

Fast smoke test with one item and no approval prompt:

```powershell
$env:EVAL_LLM_PROVIDER="ollama"
$env:EVAL_AGENT_A_MODEL="qwen2.5:7b"
$env:EVAL_AGENT_B_MODEL="llama3.1:8b"
$env:EVAL_PARALLEL_ITEMS="1"
$env:EVAL_MAX_ITEMS="1"
$env:EVAL_SKIP_APPROVAL="1"
python eval_loop.py
```

Larger Ollama settings:

- Set `EVAL_LLM_PROVIDER=ollama` to use Ollama.
- Ollama local models do not use `MINIMAX_API_KEY` or another API key by default.
- `OLLAMA_BASE_URL=http://localhost:11434`
- `EVAL_AGENT_A_MODEL=qwen2.5:7b`
- `EVAL_AGENT_B_MODEL=llama3.1:8b`
- `EVAL_PARALLEL_ITEMS=4`
- `EVAL_LLM_TIMEOUT=120`
- `EVAL_LLM_SEED=42`

## Default Provider

By default, `python eval_loop.py` and `run_eval.bat` use local Ollama with `qwen2.5:3b` for both agents. Agent A and Agent B still have different evaluator and peer-review prompts in `agent.py`.

## Run With OpenAI

PowerShell example:

```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:EVAL_LLM_PROVIDER="openai"
$env:EVAL_AGENT_A_MODEL="gpt-4o-mini"
$env:EVAL_AGENT_B_MODEL="gpt-4.1-mini"
$env:EVAL_PARALLEL_ITEMS="4"
python eval_loop.py
```

OpenAI settings:

- `EVAL_LLM_PROVIDER=openai`
- `OPENAI_API_KEY` is required.
- `OPENAI_BASE_URL=https://api.openai.com/v1`
- `EVAL_AGENT_A_MODEL=gpt-4o-mini` by default when `EVAL_LLM_PROVIDER=openai`.
- `EVAL_AGENT_B_MODEL=gpt-4.1-mini` by default when `EVAL_LLM_PROVIDER=openai`.
- `EVAL_PARALLEL_ITEMS=4`
- `EVAL_LLM_TIMEOUT=120`

## Results

After a run, inspect `results/research_report.md` first. It summarizes the evidence in a fixed research format.

Key result fields in `results/eval_results.json`:

- `summary.accuracy`: consensus pass rate.
- `summary.disagreement_metrics.evaluator_disagreement_rate`: how often Agent A and Agent B made different pass/fail decisions.
- `summary.inter_rater_reliability.cohen_kappa`: chance-adjusted agreement between Agent A and Agent B pass/fail decisions.
- `summary.inter_rater_reliability.observed_agreement`: raw pass/fail agreement between Agent A and Agent B.
- `summary.disagreement_metrics.peer_review_disagreement_rate`: how often one or both peer reviews rejected the other judge's decision.
- `summary.bias_metrics.pass_rate_gap`: pass-rate gap between judges.
- `summary.bias_metrics.leniency_bias_toward`: judge that passes more often.
- `summary.bias_metrics.confidence_gap`: average confidence-score gap between judges.
- `summary.bias_metrics.category_bias`: per-category pass, disagreement, and manual-review metrics.

Interpretation guide:

- Low disagreement and low bias gaps support stronger confidence in the consensus score.
- Higher Cohen's kappa supports stronger inter-rater reliability; values below `0.61` should be treated as weaker than substantial agreement.
- High disagreement means the answers need manual inspection before claiming performance.
- A leniency gap means one judge is systematically more permissive.
- Category ranges show whether the system performs or judges inconsistently across question types.

## Limitations

- LLM judges can be wrong, overconfident, or biased.
- If both judges use the same local model, their errors may be correlated.
- Ollama model quality depends on the model, quantization, hardware, and runtime settings.
- The gold answer is assumed to be correct. Wrong or ambiguous gold answers weaken the evaluation.
- A small or imbalanced dataset cannot prove production reliability.
- Disagreement and bias metrics improve auditability, but they do not replace expert review for high-risk decisions.

## Start In VS Code

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Run:

```bash
python eval_loop.py
```

On Windows you can also double-click:

```text
run_eval.bat
```

The sample dataset currently has fewer than 50 rows, so the script prints a warning. That warning is intentional and does not stop the run.

## Customize Agents

Open `agent.py`.

- Edit `evaluate_with_agent_a(...)` to change the strict evaluator.
- Edit `evaluate_with_agent_b(...)` to change the semantic evaluator.
- Edit `review_agent_b_work(...)` to change how Agent A reviews Agent B.
- Edit `review_agent_a_work(...)` to change how Agent B reviews Agent A.
- Replace `ask_agent(question)` if your dataset does not already include `chatbot_answer`.

## Runtime Note

Each dataset row makes 4 LLM calls:

- Agent A evaluates the chatbot answer.
- Agent B evaluates the chatbot answer.
- Agent A reviews Agent B.
- Agent B reviews Agent A.

A 50-row dataset makes 200 LLM calls. Use `EVAL_MAX_ITEMS=1` for a quick smoke test.

`EVAL_PARALLEL_ITEMS` controls how many dataset rows run concurrently. Each active row can make up to 2 LLM calls at once during the judge phase and 2 LLM calls at once during the peer-review phase, so lower this value if you hit provider rate limits.
