"""Export self-evolution artifacts to training-friendly formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from open_r1.self_evolve.io import write_jsonl


REWARD_KEYS = ["answer", "budget", "process", "grounding", "consistency"]


def reward_scalar(example: Dict[str, Any]) -> float:
    """Config-driven weighted scalar recorded by score_and_route_trajectories.

    This is the ONLY scalar used to rank trajectories. There is deliberately no
    fallback to an unweighted (equal-weight) sum: if ``reward_scalar`` is
    absent the trajectory was not scored through the config pipeline and ranks
    last (0.0), which surfaces the omission rather than silently equal-weighting.
    """
    return float(example.get("reward_scalar", 0.0))


def build_sft_replay_examples(scored_examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SFT replay = positive buffer ONLY.

    Correct-but-shortcut / low-grounding trajectories are failures upstream, so
    they can never enter SFT here. The scorer keeps at most one positive per task.
    """
    replay: List[Dict[str, Any]] = []
    for example in scored_examples:
        if example.get("buffer") != "positive":
            continue
        replay.append(
            {
                "task_id": example.get("task_id"),
                "problem": example.get("problem"),
                "prompt": example.get("prompt"),
                "completion": example.get("completion"),
                "solution": example.get("completion"),
                "image": example.get("image"),
                "image_path": example.get("image_path"),
                "source_buffer": "positive",
                "reward_vector": example.get("reward_vector"),
                "reward_scalar": example.get("reward_scalar"),
                "routing_reason": example.get("routing_reason"),
                "failure_tags": example.get("failure_tags"),
            }
        )
    return replay


def _image_value(task: Dict[str, Any]) -> Any:
    if task.get("image") is not None:
        return task["image"]
    if task.get("image_path") is not None:
        return task["image_path"]
    return None


def build_grpo_task_examples(accepted_tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build JSONL records consumable by grpo_jsonl.py.

    The training script joins `image_root` with the `image` field. Absolute
    paths remain absolute under os.path.join, so both relative and absolute
    image paths are supported.
    """

    examples: List[Dict[str, Any]] = []
    for task in accepted_tasks:
        problem = str(task.get("problem") or "")
        solution = str(task.get("solution") or "")
        image_value = _image_value(task)
        human_value = problem
        if image_value is not None:
            human_value = "<image>\n" + problem

        record = {
            "task_id": task.get("task_id"),
            "base_task_id": task.get("base_task_id"),
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": solution},
            ],
            "accu_reward_method": "default",
            "self_evolve": {
                "generator": task.get("generator"),
                # Embed only the TRAINING-RELEVANT, type-stable subset of the
                # reference judgment. The full reference_judge dict carries the
                # live Reference VLM's free-form generator_feedback (e.g.
                # avoid_scene_id, which the model emits as bool OR string), and
                # mixing those types across rows breaks pyarrow type inference in
                # Dataset.from_list (grpo_jsonl). The live reward only needs
                # budget + reference reasoning steps, so we project to those.
                "reference": _sanitize_reference_for_training(task.get("reference_judge")),
                # Grounding context for the LIVE GRPO reward. Without this block
                # the live reward had no reference boxes and grounding collapsed
                # to a constant fallback (no GRPO signal from <bbox>). Carrying
                # the gold pixel boxes + image size + valid player ids lets the
                # live reward score the model-emitted <bbox> via the SAME IoU
                # definition as the offline scorer.
                "grounding": _grounding_context_for_training(task),
            },
        }
        if image_value is not None:
            record["image"] = image_value
        examples.append(record)
    return examples


def _grounding_context_for_training(task: Dict[str, Any]) -> Dict[str, Any]:
    """Project the task's grounding metadata into a type-stable training block.

    The live GRPO reward scores the model-emitted ``<bbox>`` against these gold
    pixel boxes. Keeping the image size + valid player ids here means the live
    reward uses exactly the same IoU definition as the offline scorer, and that
    a model bbox can be converted from normalized → pixel coordinates.

    All values are coerced to stable types so pyarrow inference in
    ``Dataset.from_list`` does not break across rows.
    """
    meta = task.get("metadata", {}) or {}
    comp = meta.get("comparison_data", {}) or {}
    gold = (
        meta.get("gold_evidence_boxes")
        or comp.get("gold_evidence_boxes")
        or task.get("gold_bbox")
        or []
    )

    def _box(b: Any) -> Optional[List[float]]:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                return [float(v) for v in b]
            except (TypeError, ValueError):
                return None
        return None

    gold_boxes = [bb for bb in (_box(b) for b in gold) if bb is not None]

    num_players = meta.get("num_players")
    try:
        num_players = int(num_players) if num_players is not None else 3
    except (TypeError, ValueError):
        num_players = 3

    spy_player = meta.get("spy_player")
    try:
        spy_player = int(spy_player) if spy_player is not None else None
    except (TypeError, ValueError):
        spy_player = None

    return {
        # Prefer the measured size (variants carry it); 320x240 is the CLEVR
        # replacement-render default. image_width/height let the live
        # reward convert the model's normalized bbox to pixels.
        "image_width": int((meta.get("variant") or {}).get("image_width") or 320),
        "image_height": int((meta.get("variant") or {}).get("image_height") or 240),
        "gold_evidence_boxes": gold_boxes,
        "valid_player_ids": [spy_player] if spy_player is not None else [p + 1 for p in range(num_players)],
        "spy_player": spy_player,
        "base_name": str(meta.get("base_name") or ""),
        "grounding_source_kind": "clevr_metadata_replaced_object_boxes",
    }


def _sanitize_reference_for_training(
    reference_judge: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Project the reference judgment to a type-stable, training-relevant subset.

    Drops the free-form ``generator_feedback`` block (mixed bool/str values like
    ``avoid_scene_id`` that break pyarrow) and keeps only the fields the live
    GRPO reward consumes. Every value is coerced to a stable type.
    """
    rj = reference_judge or {}

    def _opt_int(v: Any) -> Optional[int]:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    steps = rj.get("reference_reasoning_steps") or []
    if not isinstance(steps, list):
        steps = []
    return {
        "reference_source": str(rj.get("reference_source") or ""),
        "difficulty_level": str(rj.get("difficulty_level") or ""),
        "reasoning_budget_steps": _opt_int(rj.get("reasoning_budget_steps")),
        "reasoning_budget_tokens": _opt_int(rj.get("reasoning_budget_tokens")),
        "reference_reasoning_steps": [str(s) for s in steps],
    }


def write_grpo_data_config(
    output_yaml: str | Path,
    grpo_jsonl: str | Path,
    sampling_strategy: str = "all",
) -> None:
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "datasets:",
        f"  - json_path: {Path(grpo_jsonl)}",
        f"    sampling_strategy: {sampling_strategy}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_exports(
    output_dir: str | Path,
    accepted_tasks: List[Dict[str, Any]],
    scored_examples: List[Dict[str, Any]],
    scored_proposals: Optional[List[Dict[str, Any]]] = None,
    proposer_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Write the round's SFT and GRPO files.

    With self-play on, the proposals that turned out learnable join the SFT
    file, so one training pass teaches the model both roles it plays.
    """
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    solver_replay = build_sft_replay_examples(scored_examples)
    proposer_replay: List[Dict[str, Any]] = []
    if scored_proposals:
        from open_r1.self_evolve.proposer import (
            DEFAULT_LEARNABILITY_THRESHOLD,
            build_proposer_sft_examples,
        )
        proposer_replay = build_proposer_sft_examples(
            scored_proposals,
            threshold=(proposer_threshold
                       if proposer_threshold is not None
                       else DEFAULT_LEARNABILITY_THRESHOLD),
        )
    sft_replay = solver_replay + proposer_replay
    grpo_tasks = build_grpo_task_examples(accepted_tasks)

    sft_path = export_dir / "sft_replay.jsonl"
    grpo_path = export_dir / "grpo_tasks.jsonl"
    grpo_yaml = export_dir / "grpo_tasks.yaml"

    write_jsonl(sft_path, sft_replay)
    write_jsonl(grpo_path, grpo_tasks)
    write_grpo_data_config(grpo_yaml, grpo_path)

    return {
        "sft_replay_jsonl": str(sft_path),
        "grpo_tasks_jsonl": str(grpo_path),
        "grpo_data_config": str(grpo_yaml),
        "num_sft_replay": len(sft_replay),
        "num_sft_solver": len(solver_replay),
        "num_sft_proposer": len(proposer_replay),
        "num_grpo_tasks": len(grpo_tasks),
    }
