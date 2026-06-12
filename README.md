# Remember, Don't Re-read

**A stateful ReAct agent that cuts Karpathy-style AutoResearch token cost by ~10× without sacrificing performance.**

[Karpathy's *autoresearch* pattern](https://github.com/karpathy/autoresearch) lets an LLM run ML experiments on its own by iteratively editing a training script to optimize a target metric. It's elegant but **stateless**: every iteration rebuilds the full experiment history from scratch, so token cost grows linearly per step (`O(n)`) and quadratically overall (`O(n²)`).

This repo reformulates that loop as a **stateful ReAct agent** built on LangGraph. Experiment history, strategic reasoning, and convergence tracking live in a typed, persistent state graph and are passed to the model through a tool-calling interface instead of an ever-growing prompt. The result is `O(1)` token cost per iteration, so experiment sequences can run arbitrarily long without prompt truncation or summarization.

## Key result

On an XGBoost hyperparameter-optimization benchmark (UCI Covertype, 15 iterations, 3 seeds):

| | Stateless (Karpathy) | Stateful (ReAct) |
|---|---|---|
| Tokens used | 24,465 | **2,492** (~90% fewer) |
| Per-iteration cost | `O(n)` | `O(1)` |
| Optimization quality | comparable | comparable |
| Wall-clock time | 116 s | 373 s (3.2× slower) |

The token reduction is structural, not a prompt-engineering trick: the stateless agent re-reads the whole history each step, while the stateful agent works within a fixed-size conversation window. The trade-off is wall-clock time, since the ReAct loop makes several model round-trips per iteration.

## How it works

The agent is a LangGraph state graph with three node types: **Reason** (invokes the LLM to produce reasoning + tool calls), **Tools** (executes tool calls and returns observations), and **Check** (evaluates convergence without involving the LLM). Tools wrap the training workflow — exporting/modifying the training notebook with guardrails, submitting training jobs, and querying results.

## Paper

Full method, benchmark, and analysis are in the accompanying paper (`main.tex`). Author: Faramarz Jabbarvaziri.
