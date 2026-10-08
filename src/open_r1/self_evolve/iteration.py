"""End-to-end offline self-evolution iteration."""

from __future__ import annotations

import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_r1.self_evolve.buffer import route_with_config
from open_r1.self_evolve.failure_tags import assign_failure_tags
from open_r1.self_evolve.io import read_jsonl
from open_r1.self_evolve.protocol_fingerprint import self_evolve_protocol_fingerprint
from open_r1.self_evolve.reward_config import RewardConfig, load_reward_config
from open_r1.self_evolve.rewards import answer_reward, compute_group_reward_vectors, is_format_valid, parse_structured_answer
from open_r1.self_evolve.visual_facts import gold_changes_from_task


REWARD_KEYS = ["answer", "budget", "process", "grounding", "consistency"]


def _answer_judge_enabled() -> bool:
    """Answer LLM judge is opt-in via env (set by the loop driver).

    ``SELF_EVOLVE_ANSWER_JUDGE=1`` turns on the layer-2 judge fallback. Off by
    default so offline scoring stays rule-only and free.
    """
    import os
    return os.environ.get("SELF_EVOLVE_ANSWER_JUDGE", "").strip() in ("1", "true", "True")


def _answer_judge_dry_run() -> bool:
    """When the judge is enabled, dry-run (no API) unless explicitly allowed.

    ``SELF_EVOLVE_ANSWER_JUDGE_LIVE=1`` permits real API calls; otherwise the
    judge runs in dry-run and reports blocked_reason=no_api_or_dry_run.
    """
    import os
    return os.environ.get("SELF_EVOLVE_ANSWER_JUDGE_LIVE", "").strip() not in ("1", "true", "True")


def _answer_judge_force() -> bool:
    """Experiment probe: force the answer judge on EVERY trajectory.

    ``SELF_EVOLVE_ANSWER_JUDGE_FORCE=1`` invokes the judge even on exact-match
    successes so a small live probe can demonstrate real judge involvement.
    Off by default — production scoring keeps the fallback-only behaviour.
    """
    import os
    return os.environ.get("SELF_EVOLVE_ANSWER_JUDGE_FORCE", "").strip() in ("1", "true", "True")


def _answer_judge_concurrency() -> int:
    import os
    try:
        raw = os.environ.get("SELF_EVOLVE_ANSWER_JUDGE_CONCURRENCY", "16").strip() or "16"
        return max(1, int(raw))
    except ValueError:
        return 16


def _answer_judge_sample_n() -> int:
    """Experiment probe: invoke the judge on the first N trajectories per group.

    ``SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N=4`` => first 4 trajectories of each task
    group get a (live, if enabled) judge call regardless of exact match. 0 = off.
    """
    import os
    try:
        return max(0, int(os.environ.get("SELF_EVOLVE_ANSWER_JUDGE_SAMPLE_N", "0").strip() or "0"))
    except ValueError:
        return 0


def _build_task_metadata(
    example: Dict[str, Any],
    default_max_reasoning_words: int,
) -> Dict[str, Any]:
    """Collect per-task metadata for reward computation.

    Pulls from reference_judge, task metadata, and comparison_data.
    """
    meta = dict(example.get("metadata", {}))
    self_evolve_task = meta.get("self_evolve_task", {}) or {}

    # Reference judge info
    ref_judge = self_evolve_task.get("reference_judge", {}) or {}
    if not ref_judge:
        ref_judge = meta.get("reference_judge", {}) or {}

    task_metadata: Dict[str, Any] = {
        "reasoning_budget_steps": ref_judge.get("reasoning_budget_steps"),
        "reasoning_budget_tokens": ref_judge.get("reasoning_budget_tokens"),
        "budget_source": ref_judge.get("budget_source"),
        "difficulty_level": ref_judge.get("difficulty_level"),
        "expected_reasoning_steps": ref_judge.get("expected_reasoning_steps"),
        "reference_reasoning_steps": ref_judge.get("reference_reasoning_steps"),
        "required_visual_evidence": ref_judge.get("required_visual_evidence"),
    }

    # Remove None values
    task_metadata = {k: v for k, v in task_metadata.items() if v is not None}

    # Also carry over grounding and box info from metadata
    comp_data = meta.get("comparison_data", {}) or {}
    if comp_data.get("gold_evidence_boxes"):
        task_metadata["gold_evidence_boxes"] = comp_data["gold_evidence_boxes"]
    if meta.get("spy_player") is not None:
        try:
            task_metadata["valid_player_ids"] = [int(meta["spy_player"])]
        except (TypeError, ValueError):
            pass
    variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
    task_metadata["image_width"] = int(variant.get("image_width") or 320)
    task_metadata["image_height"] = int(variant.get("image_height") or 240)

    # Offline buffer routing must score the same visual certificate as live
    # GRPO. The trajectory carries its source task, including the retained
    # variant indices; derive the gold transitions from that exact task.
    fact_task = self_evolve_task if self_evolve_task else example
    task_metadata["gold_visual_changes"] = gold_changes_from_task(fact_task)

    return task_metadata


def _extract_predicted_boxes(example: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Extract predicted evidence boxes from a trajectory example."""
    boxes = example.get("predicted_evidence_boxes")
    if boxes is not None:
        return boxes
    meta = example.get("metadata", {}) or {}
    return meta.get("predicted_evidence_boxes")


def _extract_gold_boxes(example: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Extract gold evidence boxes from a task example."""
    boxes = example.get("gold_evidence_boxes")
    if boxes is not None:
        return boxes
    meta = example.get("metadata", {}) or {}
    comp_data = meta.get("comparison_data", {}) or {}
    boxes = comp_data.get("gold_evidence_boxes") or meta.get("gold_evidence_boxes")
    return boxes


def group_by_task(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        task_id = example.get("task_id")
        if task_id is None:
            raise ValueError("Each trajectory must contain task_id.")
        groups[str(task_id)].append(example)
    return groups


def _reasoning_budget(task_examples: List[Dict[str, Any]], default: int) -> int:
    for example in task_examples:
        task = example.get("metadata", {}).get("self_evolve_task", {})
        budget = task.get("reference_judge", {}).get("reasoning_budget_tokens")
        if budget is not None:
            return int(budget)
    return default


def _resolve_grounding_scores(
    task_examples: List[Dict[str, Any]],
    dataset_root: Optional[str],
) -> List[Optional[float]]:
    """Precomputed grounding scores only.

    No scene-metadata proxy: a completion without a parseable box scores 0,
    exactly as the live GRPO reward scores it. A proxy here would route
    box-free rollouts into the positive/SFT buffer on credit the training
    reward never paid — and for edited variants it would score against the
    base scene's full gold set rather than the variant's.
    """
    return [
        float(ex["grounding_score"]) if ex.get("grounding_score") is not None else None
        for ex in task_examples
    ]


def _score_task(
    task_examples: List[Dict[str, Any]],
    reward_config: RewardConfig,
    dataset_root: Optional[str],
    default_max_reasoning_words: int,
) -> List[Dict[str, Any]]:
    completions = [example.get("completion", "") for example in task_examples]
    solution = task_examples[0].get("solution", "")
    reference_reasoning = task_examples[0].get("reference_reasoning")
    max_reasoning_words = _reasoning_budget(task_examples, default_max_reasoning_words)

    # Resolve grounding per trajectory: use a precomputed grounding_score
    # when present, else compute from the CLEVR scene metadata (shared with
    # the live reward). None stays None only when no context is available.
    grounding_scores = _resolve_grounding_scores(task_examples, dataset_root)

    # Build task-level metadata for per-task budget and grounding
    first_example = task_examples[0]
    task_metadata = _build_task_metadata(first_example, max_reasoning_words)

    # Collect per-trajectory boxes if present
    pred_boxes_per_traj = [
        _extract_predicted_boxes(ex) for ex in task_examples
    ]
    if reward_config.process_cfg["mode"] == "visual_facts":
        # The live GRPO reward only reads boxes from the completion. External
        # or keyword-derived trajectory fields would otherwise let offline
        # routing select a positive that earned no grounding during training.
        grounding_scores = [None] * len(task_examples)
        pred_boxes_per_traj = [None] * len(task_examples)
    gold_boxes = _extract_gold_boxes(first_example)

    reward_vectors = compute_group_reward_vectors(
        completions=completions,
        solution=solution,
        reference_reasoning=reference_reasoning,
        grounding_scores=grounding_scores,
        max_reasoning_words=max_reasoning_words,
        task_metadata=task_metadata,
        gold_evidence_boxes=gold_boxes,
        predicted_evidence_boxes_per_traj=pred_boxes_per_traj,
        include_details=True,
        enable_answer_judge=_answer_judge_enabled(),
        answer_judge_dry_run=_answer_judge_dry_run(),
        answer_judge_force=_answer_judge_force(),
        answer_judge_sample_n=_answer_judge_sample_n(),
        problem=task_examples[0].get("problem") or task_examples[0].get("prompt"),
        reward_config=reward_config,
    )

    task_scored: List[Dict[str, Any]] = []
    for example, reward_vector in zip(task_examples, reward_vectors):
        failure_tags = assign_failure_tags(reward_vector)
        details = reward_vector.get("reward_details", {})
        buffer_name, routing_reason = route_with_config(
            reward_vector,
            failure_tags,
            reward_config,
            reward_details=details,
        )

        # Same gate inputs as the live reward, so the offline scalar
        # matches what training optimized.
        _pred = parse_structured_answer(example.get("completion") or "")
        _gold = parse_structured_answer(solution or "")
        _spy_ok = (None if _gold.get("spy") is None
                   else _pred.get("spy") == _gold.get("spy"))
        audit = reward_config.audit_fields(
            reward_vector,
            spy_correct=_spy_ok,
            # The live GRPO gate uses the exact answer string, even when the
            # answer dimension awards full field-wise credit for a harmless
            # formatting variant. Keep offline replay scalarization identical.
            answer_correct=answer_reward(example.get("completion") or "", solution or "") >= 1.0,
            format_valid=is_format_valid(example.get("completion") or ""),
        )
        reward_scalar = audit["reward_scalar_used"]
        if isinstance(details, dict):
            details.update(audit)
            details["routing_decision"] = buffer_name
            details["routing_reason"] = routing_reason

        scored_example = dict(example)
        scored_example["reward_vector"] = reward_vector
        scored_example["failure_tags"] = failure_tags
        scored_example["buffer"] = buffer_name
        scored_example["routing_reason"] = routing_reason
        scored_example["reward_scalar"] = reward_scalar
        scored_example["reward_config_path"] = reward_config.path
        task_scored.append(scored_example)

    # SFT keeps at most one high-quality completion per task. This avoids
    # replay being dominated by near-duplicate generations from an easy
    # prompt while leaving every trajectory available for audit.
    positives = [ex for ex in task_scored if ex.get("buffer") == "positive"]
    if len(positives) > 1:
        positives.sort(key=lambda ex: float(ex.get("reward_scalar", 0.0)), reverse=True)
        for ex in positives[1:]:
            ex["buffer"] = "unused"
            ex["selection_reason"] = "positive_not_top1_for_task"

    return task_scored


def score_and_route_trajectories(
    trajectories: List[Dict[str, Any]],
    default_max_reasoning_words: Optional[int] = None,
    include_details: bool = False,
    reward_config: Optional[RewardConfig] = None,
    dataset_root: Optional[str] = None,
    stream_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Score trajectories and route them into buffers.

    Routing uses the reward VECTOR + thresholds from ``reward_config`` (loaded
    from the default config when not supplied). The config-driven weighted
    scalar is recorded per trajectory and into ``reward_details`` for audit; it
    is NOT used as a routing gate.

    ``dataset_root`` enables offline grounding: when a trajectory has no
    precomputed ``grounding_score``, grounding is computed from the CLEVR scene
    metadata (gold replaced-object boxes vs boxes parsed from the completion),
    using the SAME definition as the live GRPO reward. Without it, grounding
    falls back to whatever ``grounding_score`` the record carries (often None →
    0.0), which starves the positive buffer.
    """
    if reward_config is None:
        reward_config = load_reward_config()
    if default_max_reasoning_words is None:
        default_max_reasoning_words = int(reward_config.budget_cfg["fallback_tokens"])

    # reward_details are required to read format_valid / shortcut_detected for
    # routing, so force them on regardless of the caller's flag.
    scored: List[Dict[str, Any]] = []

    # --- Streaming/resume mode (opt-in via stream_path) --------------------
    # When stream_path is set, each task's scored rows are appended to the file
    # AS SOON AS that task is judged (per-task GPT-4o answer-judge is the
    # expensive step). A crash mid-stage keeps every already-judged task on
    # disk; a rerun skips those task_ids and only re-judges the remainder — no
    # duplicate API spend.
    stream_fp = None
    done_task_ids: set[str] = set()
    grouped = group_by_task(trajectories)
    if stream_path is not None:
        stream_path = Path(stream_path)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path = stream_path.with_name(stream_path.name + ".protocol.json")
        input_hash = hashlib.sha256(
            json.dumps(trajectories, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        expected_protocol = {
            "code_and_reward_sha256": self_evolve_protocol_fingerprint(reward_config.path),
            "input_sha256": input_hash,
            "dataset_root": str(Path(dataset_root).resolve()) if dataset_root else None,
            "answer_judge": {
                "enabled": _answer_judge_enabled(),
                "dry_run": _answer_judge_dry_run(),
                "force": _answer_judge_force(),
                "sample_n": _answer_judge_sample_n(),
            },
        }
        if stream_path.is_file() and stream_path.stat().st_size > 0:
            try:
                saved_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"{stream_path} has scored rows without a readable protocol record; "
                    "use a fresh run directory or disable resume."
                ) from exc
            if saved_protocol != expected_protocol:
                raise RuntimeError(
                    f"{stream_path} was scored under another task/reward/code protocol; "
                    "use a fresh run directory or disable resume."
                )
            stream_counts: Dict[str, int] = defaultdict(int)
            for _row in read_jsonl(stream_path):
                _tid = _row.get("task_id")
                if _tid is not None:
                    done_task_ids.add(str(_tid))
                    stream_counts[str(_tid)] += 1
                scored.append(_row)
            if any(task_id not in grouped or count != len(grouped[task_id])
                   for task_id, count in stream_counts.items()) or len(scored) != sum(stream_counts.values()):
                raise RuntimeError(
                    f"{stream_path} has an incomplete or duplicate task group; "
                    "use a fresh run directory or disable resume."
                )
        protocol_path.write_text(json.dumps(expected_protocol, indent=2), encoding="utf-8")
        stream_fp = stream_path.open("a", encoding="utf-8")

    pending = [
        (str(_task_id), task_examples)
        for _task_id, task_examples in sorted(grouped.items())
        if not (stream_fp is not None and str(_task_id) in done_task_ids)
    ]

    try:
        if pending:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from tqdm.auto import tqdm

            workers = max(1, min(_answer_judge_concurrency(), len(pending)))
            task_scored_by_id: Dict[str, List[Dict[str, Any]]] = {}
            with tqdm(total=len(pending), desc="reward judge", unit="task") as bar:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            _score_task,
                            task_examples,
                            reward_config,
                            dataset_root,
                            default_max_reasoning_words,
                        ): task_id
                        for task_id, task_examples in pending
                    }
                    for future in as_completed(futures):
                        task_id = futures[future]
                        task_scored = future.result()
                        task_scored_by_id[task_id] = task_scored
                        # Persist this task's rows the moment it's judged. flush +
                        # fsync so a hard kill can't lose an already-paid-for judgment.
                        if stream_fp is not None:
                            for _row in task_scored:
                                stream_fp.write(json.dumps(_row, ensure_ascii=False) + "\n")
                            stream_fp.flush()
                            os.fsync(stream_fp.fileno())
                        bar.update(1)
            for task_id, _task_examples in pending:
                scored.extend(task_scored_by_id[task_id])
    finally:
        if stream_fp is not None:
            stream_fp.close()

    return scored


def split_buffers(scored_examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buffers = {"positive": [], "failure": [], "unused": []}
    for example in scored_examples:
        buffer_name = example.get("buffer", "failure")
        buffers.setdefault(buffer_name, []).append(example)
    return buffers
