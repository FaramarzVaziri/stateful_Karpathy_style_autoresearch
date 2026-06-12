# Remember, Don't Re-read

**A stateful ReAct agent that cuts autonomous-experimentation token cost by 2–10× without sacrificing performance.**

[Karpathy's *autoresearch* pattern](https://github.com/karpathy/autoresearch) lets an LLM run experiments on its own by iteratively editing code to optimize a target metric. It's elegant but **stateless**: every iteration rebuilds the full experiment history from scratch, so token cost grows linearly per step (`O(n)`) and quadratically overall (`O(n²)`).

This repo reformulates that loop as a **stateful ReAct agent** built on LangGraph. Experiment history, strategic reasoning, and convergence tracking live in a typed, persistent state graph and are passed to the model through a tool-calling interface instead of an ever-growing prompt. The LLM sees only a compact summary (recent results + current best) within a fixed 20-message window, so per-iteration token cost is `O(1)` and experiment sequences can run arbitrarily long without prompt truncation or summarization.

## Key results

Evaluated on two tasks with different observation sizes per iteration (Claude Haiku 4.5, 3 seeds each).

**Task 1 — Hyperparameter tuning** (XGBoost on UCI Covertype, 15 iterations, small ~200-token observations):

| | Stateless | Stateful | Ratio |
|---|---|---|---|
| Total tokens | 24,465 | 2,492 | **9.8× fewer** |
| Best macro F1 | 0.764 | 0.779 | comparable |
| Per-iteration cost | `O(n)` | `O(1)` | — |
| Wall-clock time | 116 s | 373 s | 3.2× slower |

**Task 2 — Code performance optimization** (40 iterations, large ~3,000-token observations carrying full source + benchmark output):

| | Stateless | Stateful | Ratio |
|---|---|---|---|
| Total tokens | 1,275,309 | 626,702 | **2.0× fewer** |
| Best speedup | 1.87× | 1.91× | comparable |
| Per-iteration cost | `O(n)` | `O(1)` | — |
| Wall-clock time | 357 s | 418 s | 1.2× slower |

The reduction is **structural, not a prompt-engineering trick**: the stateless agent re-reads the whole history each step, while the stateful agent works within a fixed-size window. How big the saving is depends on the ratio of observation size to state summary — small observations (Task 1) give a ~10× reduction, while large observations (Task 2, where the current best code must still be sent every step) give ~2×. In both cases the stateless cost is `O(n²)` total vs `O(n)` for stateful, so the gap keeps widening with more iterations. The trade-off is wall-clock time, since the ReAct loop makes several model round-trips per iteration.

## How it works

The agent is a LangGraph state graph with three node types:

- **Reason** — invokes the LLM to produce reasoning traces + tool calls.
- **Tools** — executes tool calls deterministically and returns observations.
- **Check** — evaluates convergence (metric threshold, iteration budget, or agent judgment) without involving the LLM.

Tools wrap the training workflow — exporting/modifying the training notebook with guardrails (it rejects edits that strip out the evaluation or logging calls), submitting training jobs, and querying results.

## Paper

Full method, both benchmarks, and analysis are in the accompanying paper (`main.tex`). Author: Faramarz Jabbarvaziri.
