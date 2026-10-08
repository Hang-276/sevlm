# Reward and task generation

Three things live on this page: how the reward is computed, how tasks get
generated, and which numbers to watch during training. The config is
`configs/reward/reward_weights.json`. If you want to compare against the old
scoring (whole-string match, no gate, recall grounding with a free format
bonus), run `experiments/ablation/reward_component.sh old_reward` — it patches
the current config on the fly, no separate file needed.

## How the reward is computed

### Optional verified visual facts experiment

`experiments/main/ours_visual_facts.sh` switches the process dimension to a
verifiable attribute-change certificate. For every changed attribute, the
solver emits `<change>color:purple->brown</change>` (the first value belongs to
the common images, the second to the spy image). The gold multiset is built
from each task's CLEVR scene and retained object indices. Process credit is
75% multiset F1 over these claims and 25% of the existing structure score;
wrong pairings, duplicate claims, malformed tags and omissions reduce it.
The outcome gate still requires the spy answer to be correct. The experimental
reward weights are in `configs/reward/reward_visual_facts.json`; the original
`reward_weights.json` and prompt remain the default.
Only trajectories with a complete fact certificate (`visual_facts_score=1`),
plus the usual answer/box/format checks, enter the solver's positive replay
buffer. Partial certificates can still provide graded GRPO signal.

The experiment exports up to 64 deterministic, metadata-verified certificate
examples per round for SFT. It also exports up to 64 single-image questions
about the unedited scene: object count, left/right position, and depth order.
The latter use unique object descriptions and large spatial margins to avoid
ambiguous targets. These sources let SFT run even when no sampled solver
trajectory qualifies for the positive buffer; proposer-only records still do
not. Set `SELF_EVOLVE_ORACLE_SFT_MAX=0` and/or `SELF_EVOLVE_SCENE_QA_MAX=0` to
ablate them. Scene metadata supplies reward and SFT targets, never the model
prompt. The certificate checks which attributes changed, but does not yet bind
each claim to a particular object box; the independent box-F1 dimension remains
active.

Weighted sum over five dimensions; the weights and what each dimension means
all come from the reward config:

- **answer** — `spy` and `changed_attributes` are scored as separate fields
  (with tolerant parsing). Scoring the whole string makes the two sub-answers
  multiplicative: when the count collapses, the credit for getting the spy
  right vanishes with it, and the model can't tell which part it got wrong.
- **grounding** — box-F1. No unconditional format bonus (otherwise emitting
  any syntactically valid box already pays), and extra boxes cost precision
  (otherwise tiling the whole image is a winning move). Both the `<bbox>` tag
  and Qwen's native `bbox_2d` pixel coordinates are accepted.
- **consistency** — does the trajectory agree with itself: box count within
  the claimed change count, box player matches the answered spy, boxes not
  duplicated. We don't use "share of the group agreeing on the modal answer"
  because that value is constant within a rollout group — subtract the group
  mean and its GRPO gradient is exactly zero — and it rewards the group
  collapsing onto one answer, the opposite of the rollout diversity we want.
- **process / budget** — reasoning structure and length, small weights.

Within `answer`, `spy` carries more weight than `changed_attributes`. Weight a
field by how reliably it can be answered: a field that cannot be, dominating
the sum, makes answering its label prior the winning policy. The pass rate that
drives the curriculum keys on `spy` alone, the same field as the outcome gate —
requiring every field would put the pass rate at 0 whenever any one of them is
unreliable, which empties the seed list and stops the generator editing.

**The outcome gate.** grounding / process / consistency only pay when the
rollout actually identified the odd image (spy correct). Without this, farming
format credit on unsolvable tasks is a stable strategy — the `(r-mean)/std`
normalization re-inflates whatever noise is left in a saturated group back to
full gradient strength. The gate keys on spy rather than the full answer
because spy accuracy saturates early, so grounding keeps getting dense signal.

`components.grounding.warmup_steps` sets when grounding starts paying
(default 0, i.e. from the first step — once a shortcut has settled in, adding
pressure later works much worse). The timing sweep is
`experiments/sensitivity/grounding_timing.sh`.

Offline scoring (buffer routing, picking SFT samples) reads the same config
and the same definitions as live training, gate included — otherwise the
buffers get sorted by one reward while training optimizes another. A rollout
with no parseable box scores 0 on grounding, full stop; there's no
scene-metadata stand-in.

## Self-play: the model sets its own puzzles

`--self-play` (on by default via `SELF_PLAY=1`) hands task construction to the
model itself. Given a scene's two renders and the list of its changed objects,
the model picks which of them stay changed:

```
<think> ... </think>
<keep>[0,2]</keep>
```

Splicing turns that pick into a task, and the gold label is the attribute
arithmetic over exactly the kept objects. So the proposing side controls how
hard the task is and nothing else — it cannot make a task wrong, which is what
lets the two roles share one model without the reward becoming circular.

**What the proposer gets paid.** After the solver has rolled out, each proposal
is scored by what actually happened on the task it produced:
`4p(1-p)` where `p` is the solver's pass rate. That peaks at a coin flip and is
zero at both ends: a task nobody solves and a task everybody solves are equally
worthless. Proposals clearing `PROPOSER_LEARNABILITY_THRESHOLD` (0.5, so a pass
rate roughly in 0.15–0.85) join the SFT file, so the model is taught to repeat
the picks that turned out learnable. The solver keeps training with GRPO. One
model, both roles, and the frontier moves on its own as the solver improves.

**Why the two sides train differently.** The solver answers a question with a
verifiable answer, so it gets GRPO — dense, on-policy, group-relative. The
proposer's pick can only be judged *after* the solver has run, so its signal is
delayed by construction. Scoring a fresh pick inside the trainer would mean
predicting how the solver would do on it, and a predicted signal is exactly
what this method refuses everywhere else. So the proposer learns by keeping the
picks that measurably worked — no surrogate, no reward model, nothing predicted.
That asymmetry is deliberate.

**Every pick gets answered.** Proposals take edit slots, so the number of scenes
proposed over is derived from the edit budget rather than set independently.
A proposal only scores 0 when the solver actually did badly on it, never
because there was no slot left to build it in.

**It costs no extra rollouts.** Each proposal takes one of the round's task
slots, so `PROPOSER_GROUP_SIZE` proposals for a scene become that scene's
group of tasks. We changed who picks the tasks, not how many rollouts they get.

Numbers to watch in `failure_profile.json` under `proposer`:

- `parse_rate` — proposals that came back in the right format
- `mean_learnability` — how well the proposer is aiming
- `distinct_subsets` / `subset_diversity` — **a proposer that collapses onto one
  pick has stopped being an opponent**; this is where you see it first
- `num_counterfactual_pairs` — proposals that landed one object apart, which
  give the counterfactual metric for free

The ablation that isolates this is `ablation/self_play.sh off`: everything else
stays, only the picking goes back to the regret heuristic below.

## How tasks get generated

Every task already gets G rollouts, and `max_i r_i − mean_i r_i` over that
group is the task's regret — free of charge. Split tasks by pass rate:

- pass rate 0: no usable signal yet — edit it easier and send it back
- pass rate in between: there's gradient; the higher the regret, the more
  headroom — train on it, and keep it as an editing seed
- pass rate 1: nothing left to learn — retire it

This is the same quantity the trainer experiences as advantage collapse: a
group with zero spread is a group with zero gradient, so a generator that
produces fewer of those is directly reducing the collapse rate. It also means
solvability screening needs no external model — a task nobody can solve *is*
the pass-rate-0 group.

**Editing** (`task_editor.py`): `*_original.png` and `*_modified.png` are two
renders of the same scene from the same camera, pixel-aligned. Pasting one
replaced object's original patch back into the modified image undoes exactly
that object's change, and the gold count drops by exactly its attribute count.
A scene with K replaced objects yields 2^K−1 variants, every label computed
rather than estimated. If undoing one object would touch a kept object's
area, that variant is dropped — never ship an image that contradicts its
label; fresh sampling fills the freed slot.

Two things fall out of this:

- The label distribution becomes a free variable instead of whatever the image
  pool happens to contain. Difficulty is estimated from attribute salience,
  apparent object size and scene clutter — orthogonal to the label — and the
  generation report's `difficulty_label_correlation` watches that it stays so.
- Two variants one object apart form a pair (same spy slot, built and dropped
  together). A policy that answers from the label prior gives both members the
  same answer, so its difference is zero by construction.
  `counterfactual_sensitivity` = P(answer moved | input moved by one object) —
  a metric accuracy can't fake.

**One thing to check before trusting the splicing**: rectangular patches can
leave shadow seams, and a model might learn the seam instead of the object.
The control: train on spliced data only, evaluate on unspliced originals — a
big drop means it learned the seam. Or just eyeball 20 images from
`variant_cache/` first.

## The GRPO side

- Advantages subtract the group mean but don't divide by std
  (`GRPO_SCALE_REWARDS=False`) — dividing re-inflates format noise in
  saturated groups to full strength.
- Degenerate groups just get their advantages zeroed (`mask_degenerate`)
  rather than resampling the whole batch — a collapsed prompt stays collapsed
  no matter how often you resample, and batch-level retries throw away the
  healthy groups too.
- Rollouts cut off at `max_completion_length` (no EOS) get zero advantage:
  their low reward reflects the length cap, not the reasoning. Length is 1024;
  a five-image comparison needs that much, and anything shorter truncates the
  `<bbox>` block first.
- The offline solver's decode length, temperature and image resolution match
  GRPO training. All these knobs live in `configs/train_defaults.sh`.

## What to watch during training

The aggregate reward curve hides collapse; these don't:

- `P(pred_ca = mode)` — computed from `pred_changed_attributes` in the
  breakdown; drifting toward 1 means the count collapsed to a constant
- per-field accuracy — spy and count separately, in `answer_fields`
- `counterfactual_sensitivity` — printed every iteration and written into
  `failure_profile.json`; a prior-following policy scores 0 here
- `advantage_collapse_rate` / `mean_regret` — in `failure_profile.json` under
  `solvability`; the former should fall across iterations
- `mean IoU | bbox_valid` and `num_pred_boxes` — catch format-farming and
  box-spamming respectively
- `truncated_fraction` — near 0 means the length budget is enough
- `difficulty_label_correlation` — away from 0 means difficulty is leaking
  the answer again

After touching anything reward-related, run both:

```bash
python local_scripts/self_evolve/reward_alignment/test_rewards.py
python local_scripts/self_evolve/reward_alignment/audit_reward_hackability.py \
  --reward-config local_scripts/self_evolve/configs/reward/reward_weights.json \
  --max-blind-reward 0.05
```

There are three more tools in the same directory, all standalone and all
without API calls:

```bash
# Is the reward config itself valid (weights sum, dims present, no silent
# equal-weight fallback)?
python local_scripts/self_evolve/reward_alignment/check_reward_config.py

# Re-score existing trajectories under the current config, without retraining.
python .../offline_recompute_reward.py <run>/iter_000/raw_solver_trajectories.jsonl

# Grounding + group-size statistics read off a finished run.
python .../scan_grounding_numgen_stats.py <run_dir>
```

The hackability audit scores a policy that gets every answer wrong but formats
perfectly. That number has to stay low (about 0.03 on the current config;
the old scoring paid the same policy about 0.42 — reproducible via the
`old_reward` ablation).

## Resolution and the prompt budget

`GRPO_MIN_PIXELS` is the resolution the solver sees. Visual tokens are 28x28
pixels each, so `num_players` images at that resolution must fit inside
`GRPO_MAX_PROMPT_LEN` — otherwise the images are truncated and the model
answers about pictures it never saw. The loop checks the combination at launch
and warns when it does not fit.

`experiments/analysis/perception_probe.sh` reports base-model accuracy per
resolution against the always-answer-the-mode prior. Use it to pick
`GRPO_MIN_PIXELS` for your model and image pool; the script explains how to
read its output.

## Training stages

Two stages: SFT replay and GRPO (`STAGES` controls which, executed sft→grpo).
No DPO: this task's prompt text is identical across all tasks, so a text-only
preference pair can't tell tasks apart — the only thing it could learn is
style — and the signal duplicates GRPO anyway (same reward vector, which GRPO
already uses on-policy, with images). The preference-style ablation is
`STAGES=grpo` vs `STAGES=sft,grpo`.

Early iterations may have an empty positive buffer (positives need a correct
answer *and* a qualifying box). SFT then skips itself with a loud log line
and GRPO proceeds from the round model.

Training needs no API at all: solvability comes from the solver's own
rollouts, task screening falls back to local rules when the external judge is
off, and the answer judge defaults to not calling anything. Only the
evaluation MCQ judge wants a key — or use `JUDGE=exact_matching` to stay
fully local.
