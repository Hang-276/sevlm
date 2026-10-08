# Evaluation

The current protocol compares our own models on the same nine benchmarks with
the same VLMEvalKit implementation. Published Vision-Zero numbers are context,
not directly comparable without matching its full evaluation protocol.

Run everything from the repo root:

```bash
cd <repo root>
```

## What Vision-Zero does

From the Vision-Zero paper (arXiv 2509.25541) and its repo:

| | |
|---|---|
| Framework | **VLMEvalKit** (open-compass) — the paper states all models were evaluated with it |
| Philosophy | train label-free on a source domain (CLEVR / Chart / RealWorld), then measure **OOD transfer** on general benchmarks. Scores on the training distribution are **not** reported |
| Backbones | Qwen2.5-VL-7B, InternVL3-8B, InternVL3-14B |
| Compared against | R1-OneVision-7B, MM-Eureka-Qwen-7B, VLAA-Thinker-7B, OpenVLThinker-7B, ViGaL, plus GPT-4o |
| Metric | each benchmark's official accuracy, plus a group mean |
| Decoding | not established here; do not infer it from VLMEvalKit defaults |

Their benchmark suite, 13 in two groups:

- **OCR / Chart / Doc**: AI2D, ChartQA, TextVQA, DocVQA, InfoVQA, OCRBench, SEEDBench2
- **Vision-centric**: RealWorldQA, MMVP, MMStar, BLINK, MuirBench, CRPE

## Why the same protocol fits us

1. **Same backbone, same source domain** — we also train Qwen2.5-VL-7B on
   CLEVR, so `base vs Vision-Zero vs ours` is a fair three-way table.
2. **Evaluation is decoupled from training** — VLMEvalKit only measures the
   final model's OOD ability. It does not care whether the training loop was
   self-play or anything else, so changing the training mechanism never
   invalidates the evaluation.
3. **The artifacts plug straight in** — the merged full model directory that
   carry-forward produces is exactly what VLMEvalKit's Qwen2.5-VL loader
   expects. No conversion step.

## The settings we pin

Record these with every evaluation run, otherwise numbers from different
weeks stop being comparable:

- **Framework**: VLMEvalKit, pinned to one commit for the whole project. Note
  the commit in the results.
- **Decoding**: the current scripts pin `max_new_tokens=2048`,
  `temperature=0.01`, `top_p=0.001`, `top_k=1`, `do_sample=true` (near-greedy).
  Set `DO_SAMPLE=false` for strict greedy, always for every model together.
- **Attention**: sdpa. flash-attn is not required and sdpa is numerically
  equivalent for inference.
- **Splits and metrics**: each benchmark's official split and official metric.
  Never a custom variant.
- **MCQ answer extraction**: judged by `gpt-4o-mini` through the
  `OPENAI_BASE_URL` in `.env`. `JUDGE=exact_matching` switches to fully local
  scoring with no API.
- **Every model is run by us** under this one harness. Numbers copied from
  someone else's README are a sanity check, never a table row.

If the exact decoding settings from the Vision-Zero appendix become
available, use theirs instead and update this file.

## Models to evaluate

| label | model | where it comes from |
|---|---|---|
| `base` | Qwen2.5-VL-7B-Instruct | the base model in `paths.sh`. Must be run by us |
| `vision_zero` | Vision-Zero-Qwen-2.5-VL-7B-Clevr | reproduce it yourself so it is comparable |
| `ours` | the final round's solver | that run's merged model directory |
| `ours_no_self_play`, `ours_no_process`, ... | see `../experiments/README.md` | each exp's merged directory |

For any `ours_*`, the model path is the final iteration's merged model
directory. `main/eval.sh` resolves this for you — it merges the last round's
adapter when needed, and picks the newest checkpoint from a trainer output
directory.

## Current nine-benchmark suite

All default evaluation entries run exactly
`MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST MMMU_Pro_10c CV-Bench-2D CV-Bench-3D`.
For a cheap smoke run, override `DATASETS=MMVP`; use a distinct LABEL or finish
the missing eight before computing the nine-benchmark average.

```bash
# Optional: prefetch the first six benchmark TSVs into LMUData
bash local_scripts/self_evolve/eval/fetch_vlmeval_tsv.sh

# The base zero-shot row
bash local_scripts/self_evolve/experiments/main/base_zeroshot.sh
DATASETS="MMVP" bash .../main/base_zeroshot.sh          # single-benchmark smoke run
JUDGE=exact_matching bash .../main/base_zeroshot.sh     # no API

# After training, the shared evaluation entry
bash local_scripts/self_evolve/experiments/main/eval.sh <run_dir>
```

Results land under the evaluation work directory from `paths.sh`, one
subdirectory per model and possibly a VLMEvalKit eval-ID subdirectory.
`analysis/collect_results.sh` assembles the nine `*_acc.csv` metrics, checks
protocol compatibility, and withholds AVG(9) if any model is incomplete.

## The comparison table

Main table, all nine, every row run by us:

| Model | MMVP | MMStar | BLINK | RealWorldQA | AI2D_TEST | ChartQA_TEST | MMMU_Pro_10c | CV-Bench-2D | CV-Bench-3D | AVG(9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | | | | | | | | | | |
| vision_zero | | | | | | | | | | |
| ours | | | | | | | | | | |

The scripts for method comparisons are in
`../experiments/README.md`:

| exp | what it removes | what it answers |
|---|---|---|
| `ours` | nothing | the main result |
| `ours_no_self_play` | the model proposing its own puzzles | what the proposing side is worth |
| `ours_no_process` | the process-level reward | what the dense reward is worth |
| `grpo_baseline` | the whole loop | what the loop is worth |
| `stages_grpo` / `stages_sft` | one training stage | what each stage contributes |
| per-iteration checkpoints | — | the curve across rounds, which is what makes the gain look sustainable rather than one-off |

## Fairness requirements

Skip any of these and the results will be questioned:

1. **`base` and `vision_zero` must be run by us**, in our environment, at the
   pinned versions, with the same decoding. Published numbers are a sanity
   check only.
2. Every model goes through the **same VLMEvalKit commit and the same
   prompt/decoding**.
3. The report states the commit, the temperature, `max_new_tokens`, and each
   benchmark's split.
4. `ours_*` always uses the **final round's merged model**, and the report
   records which run and which iteration it came from.

## Sources

- Vision-Zero: https://arxiv.org/abs/2509.25541
- Official repo: https://github.com/wangqinsi1/Vision-Zero
- VLMEvalKit: https://github.com/open-compass/VLMEvalKit
