"""Export self-evolution artifacts to training-friendly formats."""

from __future__ import annotations

import os
import random
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from open_r1.self_evolve.io import write_jsonl
from open_r1.self_evolve.scene_qa import build_scene_qa_sft_examples
from open_r1.self_evolve.visual_facts import gold_changes_from_task
from open_r1.self_evolve.verified_qa_reward import VERIFIED_COUNT_QA


REWARD_KEYS = ["answer", "budget", "process", "grounding", "consistency"]
_ORACLE_ANSWER = re.compile(
    r"\s*<answer>\s*spy\s*=\s*([0-9]+)\s*;\s*"
    r"changed_attributes\s*=\s*([0-9]+)\s*</answer>\s*\Z"
)


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


def build_verified_oracle_replay_examples(
    accepted_tasks: Iterable[Dict[str, Any]], max_examples: int
) -> List[Dict[str, Any]]:
    """Export a capped, deterministic warm start with valid visual certificates."""
    if type(max_examples) is not int or max_examples <= 0:
        return []
    tasks = sorted((task for task in accepted_tasks if isinstance(task, dict)),
                   key=lambda task: (str(task.get("task_id") or ""),
                                     str(task.get("scene_path") or "")))
    rng = random.Random(1701)
    rng.shuffle(tasks)
    replay: List[Dict[str, Any]] = []
    for task in tasks:
        if len(replay) >= max_examples:
            break
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        changes = gold_changes_from_task(task)
        grounding = _grounding_context_for_training(task)
        boxes = grounding["gold_evidence_boxes"]
        spy = meta.get("spy_player")
        image = _image_value(task)
        if (not changes or not boxes or type(spy) is not int
                or not isinstance(image, (list, tuple)) or not 2 <= len(image) <= 8
                or not 1 <= spy <= len(image)):
            continue
        if "num_players" in meta and (type(meta["num_players"]) is not int
                                      or meta["num_players"] != len(image)):
            continue
        solution = task.get("solution")
        answer = _ORACLE_ANSWER.fullmatch(solution) if isinstance(solution, str) else None
        if answer is None or answer.groups() != (str(spy), str(len(changes))):
            continue
        bbox_lines = []
        for box in boxes:
            coords = ",".join(f"{v:g}" for v in box)
            bbox_lines.append(f'<bbox player="{spy}">[{coords}]</bbox>')
        change_lines = [
            f"<change>{c['attribute']}:{c['before']}->{c['after']}</change>"
            for c in changes
        ]
        completion = "\n".join([
            "<think>I compared the spy image with the other images and checked each changed attribute.</think>",
            *bbox_lines,
            *change_lines,
            solution.strip(),
        ])
        replay.append({
            "task_id": task.get("task_id"),
            "problem": task.get("problem"),
            "prompt": task.get("prompt"),
            "completion": completion,
            "solution": completion,
            "image": image,
            "image_path": image,
            "source_buffer": "verified_oracle",
        })
    return replay


def _image_value(task: Dict[str, Any]) -> Any:
    """Generator paths are authoritative when both image aliases are present."""
    if task.get("image_path") is not None:
        return task["image_path"]
    if task.get("image") is not None:
        return task["image"]
    return None


def _cap_auxiliary_sft(
    oracle_replay: List[Dict[str, Any]],
    scene_qa_replay: List[Dict[str, Any]],
    num_solver_positives: int,
    max_ratio: Optional[float],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Balance auxiliary targets, then cap them as real positives accrue.

    The zero-positive round keeps both cold-start pools intact. With positives,
    each pool gets up to half the allowed slots; spare slots from an exhausted
    pool go to the other. Source order within each pool stays deterministic.
    ``None`` preserves the older uncapped visual-facts export.
    """
    if max_ratio is None:
        return oracle_replay, scene_qa_replay
    if not math.isfinite(max_ratio) or max_ratio < 0:
        raise ValueError("SELF_EVOLVE_AUX_SFT_MAX_RATIO must be finite and nonnegative")
    if num_solver_positives <= 0:
        return oracle_replay, scene_qa_replay
    total = len(oracle_replay) + len(scene_qa_replay)
    cap = total if max_ratio >= total / num_solver_positives else math.floor(num_solver_positives * max_ratio)
    oracle_count = min(len(oracle_replay), (cap + 1) // 2)
    scene_count = min(len(scene_qa_replay), cap // 2)
    spare = cap - oracle_count - scene_count
    oracle_count += min(spare, len(oracle_replay) - oracle_count)
    spare = cap - oracle_count - scene_count
    scene_count += min(spare, len(scene_qa_replay) - scene_count)
    return oracle_replay[:oracle_count], scene_qa_replay[:scene_count]


def build_grpo_task_examples(accepted_tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build JSONL records consumable by grpo_jsonl.py.

    The training script joins `image_root` with the `image` field. Absolute
    paths remain absolute under os.path.join, so both relative and absolute
    image paths are supported.
    """

    examples: List[Dict[str, Any]] = []
    for task in accepted_tasks:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        qa_task = metadata.get("kind") == VERIFIED_COUNT_QA
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
                "task_kind": VERIFIED_COUNT_QA if qa_task else "clevr_spy",
                "verified_qa": {
                    "gold_count": metadata.get("gold_count") if qa_task else None,
                    "pair_id": str(task.get("pair_id") or "") if qa_task else "",
                    "pair_role": str(task.get("pair_role") or "") if qa_task else "",
                },
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
                # A compact, type-stable certificate target. The live reward
                # sees only these verified attribute transitions, never the
                # source scene path or other free-form scene metadata.
                "visual_facts": {
                    "gold_visual_changes": gold_changes_from_task(task),
                },
            },
        }
        if image_value is not None:
            record["image"] = image_value
        examples.append(record)
    return examples


def build_mixed_grpo_task_examples(
    accepted_tasks: List[Dict[str, Any]], fraction: float = 0.0, seed: int = 1701,
) -> List[Dict[str, Any]]:
    """Replace unpaired game rows with QA twins, preserving the prompt budget."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 0.5:
        raise ValueError("SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION must be finite and in [0, 0.5]")
    records = build_grpo_task_examples(accepted_tasks)
    if fraction == 0:
        return records
    eligible = []
    for index, task in enumerate(accepted_tasks):
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if not (task.get("pair_id") or metadata.get("pair_id")):
            eligible.append(index)
    cap = min(math.floor(len(records) * fraction), len(eligible))
    cap -= cap % 2
    if cap < 2:
        return records
    from open_r1.self_evolve.counterfactual_qa import build_counterfactual_qa_tasks

    qa_records = build_grpo_task_examples(
        build_counterfactual_qa_tasks(accepted_tasks, cap, seed=seed),
    )
    if not qa_records:
        return records
    rng = random.Random(seed)
    rng.shuffle(eligible)
    replaced = set(eligible[:len(qa_records)])
    mixed = [record for index, record in enumerate(records) if index not in replaced] + qa_records
    rng.shuffle(mixed)
    assert len(mixed) == len(records), "QA mixing must preserve the sampled task budget"
    return mixed


def _grounding_context_for_training(task: Dict[str, Any]) -> Dict[str, Any]:
    """Export finite pixel boxes and type-stable grounding context."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    comp = meta.get("comparison_data") if isinstance(meta.get("comparison_data"), dict) else {}
    variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
    width, height = variant.get("image_width", 320), variant.get("image_height", 240)
    dimensions_valid = all(type(value) is int and value > 0 for value in (width, height))
    if not dimensions_valid:
        width, height = 320, 240
    gold = (
        meta.get("gold_evidence_boxes")
        or comp.get("gold_evidence_boxes")
        or task.get("gold_bbox")
        or []
    )

    def _box(b: Any) -> Optional[List[float]]:
        if isinstance(b, (list, tuple)) and len(b) == 4 and not any(isinstance(v, bool) for v in b):
            try:
                coords = [float(v) for v in b]
                if (all(math.isfinite(v) for v in coords)
                        and 0 <= coords[0] < coords[2] <= width
                        and 0 <= coords[1] < coords[3] <= height):
                    return coords
            except (TypeError, ValueError, OverflowError):
                pass
        return None

    boxes = [_box(b) for b in gold] if isinstance(gold, (list, tuple)) else []
    gold_boxes = boxes if dimensions_valid and all(box is not None for box in boxes) else []
    image = _image_value(task)
    num_players = meta.get("num_players", len(image) if isinstance(image, (list, tuple)) else 3)
    players_valid = type(num_players) is int and 2 <= num_players <= 8
    if isinstance(image, (list, tuple)) and len(image) != num_players:
        players_valid = False
    if not players_valid:
        gold_boxes = []
    spy_player = meta.get("spy_player")
    if spy_player is not None and (not players_valid or type(spy_player) is not int
                                   or not 1 <= spy_player <= num_players):
        gold_boxes = []
        spy_player = None
        players_valid = False

    return {
        # Prefer the measured size (variants carry it); 320x240 is the CLEVR
        # replacement-render default. image_width/height let the live
        # reward convert the model's normalized bbox to pixels.
        "image_width": width,
        "image_height": height,
        "gold_evidence_boxes": gold_boxes,
        "valid_player_ids": ([spy_player] if spy_player is not None else list(range(1, num_players + 1)))
                            if players_valid else [],
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
    rj = reference_judge if isinstance(reference_judge, dict) else {}

    def _opt_int(v: Any) -> Optional[int]:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError, OverflowError):
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
    oracle_replay: List[Dict[str, Any]] = []
    scene_qa_replay: List[Dict[str, Any]] = []
    if os.environ.get("SELF_EVOLVE_VISUAL_FACTS", "").strip() == "1":
        oracle_max = max(0, int(os.environ.get("SELF_EVOLVE_ORACLE_SFT_MAX", "64")))
        oracle_replay = build_verified_oracle_replay_examples(accepted_tasks, oracle_max)
        scene_qa_max = max(0, int(os.environ.get("SELF_EVOLVE_SCENE_QA_MAX", "64")))
        scene_qa_replay = build_scene_qa_sft_examples(accepted_tasks, scene_qa_max)
        ratio_text = os.environ.get("SELF_EVOLVE_AUX_SFT_MAX_RATIO", "").strip()
        ratio = float(ratio_text) if ratio_text else None
        oracle_replay, scene_qa_replay = _cap_auxiliary_sft(
            oracle_replay, scene_qa_replay, len(solver_replay), ratio,
        )
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
    sft_replay = solver_replay + oracle_replay + scene_qa_replay + proposer_replay
    qa_fraction = float(os.environ.get("SELF_EVOLVE_COUNTERFACTUAL_QA_FRACTION", "0"))
    grpo_tasks = build_mixed_grpo_task_examples(accepted_tasks, qa_fraction)
    num_grpo_qa = sum(record["self_evolve"]["task_kind"] == VERIFIED_COUNT_QA for record in grpo_tasks)

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
        "num_sft_non_proposer": len(solver_replay) + len(oracle_replay) + len(scene_qa_replay),
        "num_sft_verified_oracle": len(oracle_replay),
        "num_sft_scene_qa": len(scene_qa_replay),
        "num_sft_proposer": len(proposer_replay),
        "num_grpo_tasks": len(grpo_tasks),
        "num_grpo_spy": len(grpo_tasks) - num_grpo_qa,
        "num_grpo_counterfactual_qa": num_grpo_qa,
        "num_grpo_qa_pairs": num_grpo_qa // 2,
        "grpo_qa_fraction": num_grpo_qa / len(grpo_tasks) if grpo_tasks else 0.0,
    }
