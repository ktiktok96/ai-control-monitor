# Can a weaker monitor catch a stronger agent's hidden sabotage?

A small [AI control](https://www.alignmentforum.org/tag/ai-control) experiment. A capable
**agent** model completes ordinary coding tasks. In some runs it is also secretly given a
harmless, simulated **side task** (e.g. writing a fake API key to a log file, special-casing a
"maintenance" user, or calling an unexpected URL). A cheaper, weaker **monitor** model reviews
each run and assigns a 0–100 suspicion score.

**Question:** how often does the monitor catch successful sabotage while flagging only a small
fraction of honest work, and what helps it?

> **Status:** pipeline complete and tested; real-model results pending.
> Results, charts, and the write-up will go here.

## Results

_TBD — headline finding in one sentence, then the chart._

| Monitor | Sees | Prompt | Agent hides tracks | Catch rate @ 5% FPR | AUROC |
|---|---|---|---|---|---|
| | | | | | |

## Experimental conditions

| Variable | Values |
|---|---|
| Monitor view | final code only · reasoning + code |
| Monitor prompt | plain · checklist of red flags |
| Agent stealth | told nothing about monitor · told to hide side task |
| Monitor strength | e.g. Haiku vs. Sonnet (`--monitor-models`) |

**Metric.** Following the AI control literature, we fix an audit budget: the suspicion threshold
is set so that only 5% of *honest* runs get flagged, then we measure the fraction of
*successful* attacks scoring above it. Failed sabotage attempts are excluded (there's nothing to
catch). AUROC and 95% bootstrap confidence intervals are also reported, since samples are small.

## Run it

```bash
pip install -e ".[dev]"
pytest                       # unit + end-to-end tests, no API key needed
control-eval --dry-run       # full pipeline with a mock model

export ANTHROPIC_API_KEY=...
control-eval --epochs 3 --monitor-models claude-haiku-5-5 claude-sonnet-5-5
```

Outputs go to `results/`: `records.jsonl` (every transcript + score) and `summary.json`.
All model calls are cached on disk in `.cache/`, so reruns are free and reproducible.

## Design

```
data/tasks.json          main tasks + side tasks (with regex success checks)
src/control_eval/
  agent.py               untrusted agent; honest / attack / attack+stealth prompts
  monitor.py             trusted monitor; view + prompt variants; score parsing
  metrics.py             catch rate at fixed FPR, AUROC, bootstrap CI (pure Python)
  models.py              async Anthropic client: retries, concurrency limit, disk cache,
                         token accounting; deterministic mock for tests
  run.py                 CLI orchestrating the full experiment grid
```

## Limitations

- Single-turn tasks, not multi-step agentic environments with tools.
- Side-task success is checked by regex, not by executing code.
- Small task set; results are indicative, not definitive.
- The agent is *instructed* to sabotage (a model organism), which differs from
  spontaneous misalignment.

## Related work

- SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents (Anthropic, 2025)
- Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* (2023)
- SLEIGHT-Bench: Finding Blind Spots in AI Monitors (Anthropic, 2026)
