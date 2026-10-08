"""Self-play: the model picks which changes stay, then solves what it picked.

The proposer sees a scene's two renders and chooses a subset of its changed
objects to keep. Splicing turns that subset into a task whose gold label is
computed from the subset, so the proposer cannot touch the label no matter what
it proposes — the only thing it controls is how hard the task is.

It is paid for landing the solver in the middle: a task nobody solves and a
task everybody solves both score zero. Since proposer and solver are the same
model, the frontier moves as the solver improves, which is the whole point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from open_r1.self_evolve.task_difficulty import ATTRS

# A proposal is worth training on when the solver landed near the middle.
# 4p(1-p) peaks at p=0.5 and is 0 at both ends; 0.5 here means a pass rate
# roughly between 0.15 and 0.85.
DEFAULT_LEARNABILITY_THRESHOLD = 0.5

PROPOSER_PROMPT = (
    "You are setting a spot-the-difference puzzle for another player.\n"
    "The first image is the original scene. The second image is the same scene "
    "after some objects were swapped.\n"
    "Exactly {num_objects} object(s) were swapped. They are numbered "
    "{index_range} and this is the complete list:\n"
    "{object_lines}\n"
    "{competence}"
    "You control two things.\n"
    "1. Which swapped objects stay swapped. Every one you leave out is put back "
    "to how it looks in the first image. You must keep at least one. Use only "
    "the numbers {index_range} — do not number any other object in the scene.\n"
    "2. How many players see the puzzle, from {min_players} to {max_players}. "
    "One player gets the swapped image and the rest get the original, so more "
    "players means more images to compare and a harder puzzle.\n"
    "A good puzzle is one the player can just barely solve: too easy and it "
    "teaches nothing, too hard and it teaches nothing either.\n"
    "Reply in EXACTLY this format:\n"
    "<think> your reasoning </think>\n"
    "<keep>[{example}]</keep>\n"
    "<players>{mid_players}</players>"
)

# The brackets are optional: a real model writes <keep>1</keep> often enough
# that rejecting it throws away a perfectly clear answer.
_KEEP_RE = re.compile(r"<keep>\s*\[?([^\]<]*?)\]?\s*</keep>",
                      re.IGNORECASE | re.DOTALL)
_PLAYERS_RE = re.compile(r"<players>\s*(\d+)\s*</players>", re.IGNORECASE)


@dataclass
class ProposerConfig:
    """Knobs for the proposing side of the loop."""

    # Proposals sampled per scene. These become that scene's task slots, so the
    # solver rollout budget is unchanged — we only choose which tasks get it.
    group_size: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 512
    learnability_threshold: float = DEFAULT_LEARNABILITY_THRESHOLD
    # Scenes to propose over per round. The rest of the round is sampled fresh.
    max_scenes: int = 8
    # Tell the proposer the solver's current solve rate. Off is the ablation
    # that asks whether aiming at the frontier needs to know where it is.
    show_competence: bool = True
    # How many players the proposer may ask for. This is its second axis: a
    # scene with two changed objects offers only three subsets, which is not
    # much of a game on its own. The spy slot stays random on purpose — it IS
    # the answer, and a proposer that could pick it would let both roles
    # settle on one constant slot and call it a win.
    min_players: int = 3
    max_players: int = 8


def changed_attributes_of(replaced_object: Dict[str, Any]) -> List[str]:
    """Which attributes actually differ between the two renders."""
    orig = replaced_object.get("original", {}) or {}
    repl = replaced_object.get("replacement", {}) or {}
    return [a for a in ATTRS if orig.get(a) != repl.get(a)]


def describe_changed_objects(scene: Dict[str, Any]) -> List[str]:
    """One line per changed object: what it looks like now, what it was, indexed."""
    replaced = (scene.get("modification", {}) or {}).get("replaced_objects") or []
    lines: List[str] = []
    for i, ro in enumerate(replaced):
        attrs = changed_attributes_of(ro)
        if not attrs:
            # Nothing about this object actually differs between the renders,
            # so keeping it would add an evidence box nobody can find. Don't
            # offer it as a choice.
            continue
        now = ro.get("replacement", {}) or {}
        before = ro.get("original", {}) or {}
        coords = now.get("pixel_coords") or before.get("pixel_coords") or []
        where = (f" at ({int(coords[0])}, {int(coords[1])})"
                 if len(coords) >= 2 else "")
        lines.append(
            f"  [{i}] the {_describe(now)}{where} in the second image — "
            f"it was {_describe(before)} in the first "
            f"({', '.join(attrs)} changed)"
        )
    return lines


def _describe(side: Dict[str, Any]) -> str:
    """A short visual description like 'small red rubber cube'."""
    parts = [str(side.get(a)) for a in ("size", "color", "material", "shape")
             if side.get(a)]
    return " ".join(parts) if parts else "object"


def competence_line(solvability: Optional[Dict[str, Any]]) -> str:
    """Tell the proposer how the solver is currently doing, so it can aim."""
    if not solvability:
        return ""
    rate = solvability.get("mean_solve_rate")
    if rate is None:
        return ""
    verified_rate = solvability.get("mean_verified_pass_rate")
    if solvability.get("num_certified_tasks") == 0:
        return (
            f"Right now the player identifies the spy in {float(rate) * 100:.0f}% "
            "of puzzles, but visual certificate success cannot be checked. "
            "Keep the difficulty steady until verified feedback is available.\n"
        )
    if verified_rate is not None:
        return (
            f"Right now the player identifies the spy in {float(rate) * 100:.0f}% "
            f"of puzzles, but completes the visual certificate (correct count, "
            f"grounded boxes, and exact visual changes) in "
            f"{float(verified_rate) * 100:.0f}%.\n"
            "Help it improve its visual evidence while keeping the puzzle's "
            "difficulty steady. Finding the spy alone is not a reason to add "
            "more changes or players.\n"
        )
    return (
        f"Right now the player solves {float(rate) * 100:.0f}% of the puzzles "
        f"it is given.\n"
    )


def build_proposer_prompt(
    scene: Dict[str, Any],
    solvability: Optional[Dict[str, Any]] = None,
    min_players: int = 3,
    max_players: int = 8,
) -> Optional[str]:
    """The prompt for one scene, or None when the scene has nothing to choose from."""
    lines = describe_changed_objects(scene)
    if len(lines) < 2:
        # With one changed object there is only one non-empty subset — no choice
        # to make, so there is nothing to learn from proposing here.
        return None
    valid = [int(l.split("]")[0].lstrip().lstrip("[")) for l in lines]
    index_range = (f"{valid[0]} to {valid[-1]}" if len(valid) > 1
                   else str(valid[0]))
    return PROPOSER_PROMPT.format(
        num_objects=len(lines),
        index_range=index_range,
        object_lines="\n".join(lines),
        competence=competence_line(solvability),
        example=",".join(str(v) for v in valid[:2]),
        min_players=min_players,
        max_players=max_players,
        mid_players=(min_players + max_players) // 2,
    )


def parse_proposal(
    completion: str,
    num_objects: int,
    min_players: int = 3,
    max_players: int = 8,
) -> Optional[Dict[str, Any]]:
    """Pull the kept indices and the player count out of a proposal.

    Returns None when the pick is unusable. A missing or out-of-range player
    count is not fatal — the subset is the part that decides the label, so we
    fall back to the middle of the range and keep the proposal.
    """
    if not completion or num_objects <= 0:
        return None
    match = _KEEP_RE.search(completion)
    if not match:
        return None
    keep: List[int] = []
    for token in re.split(r"[,\s]+", match.group(1).strip()):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            return None
        if value < 0 or value >= num_objects:
            return None
        keep.append(value)
    keep = sorted(set(keep))
    if not keep:
        return None

    players = (min_players + max_players) // 2
    pmatch = _PLAYERS_RE.search(completion)
    if pmatch:
        try:
            asked = int(pmatch.group(1).strip())
            if min_players <= asked <= max_players:
                players = asked
        except ValueError:
            pass
    return {"keep": keep, "num_players": players}


def learnability(pass_rate: Optional[float]) -> float:
    """How much a task taught the solver: 1.0 at a coin flip, 0 at either extreme."""
    if pass_rate is None:
        return 0.0
    try:
        p = float(pass_rate)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(p):
        return 0.0
    p = min(1.0, max(0.0, p))
    return round(4.0 * p * (1.0 - p), 4)


def score_proposals(
    proposals: Sequence[Dict[str, Any]],
    task_stats: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach each proposal the learnability the solver actually realized on it.

    ``task_stats`` is regret.task_statistics output, keyed by task_id. A proposal
    whose task never made it to the solver scores 0 — an unbuildable proposal is
    a bad proposal, and saying so is the point.
    """
    scored: List[Dict[str, Any]] = []
    for proposal in proposals:
        task_id = str(proposal.get("task_id") or "")
        stats = task_stats.get(task_id) or {}
        pass_rate = stats.get("pass_rate")
        record = dict(proposal)
        record["pass_rate"] = pass_rate
        record["solver_class"] = stats.get("class")
        record["regret"] = stats.get("regret")
        if stats.get("mastery_mode") == "verified_visual":
            evidence_frontier = stats.get("frontier") == "evidence"
            certified = (stats.get("gold_certificate_available") is True
                         and stats.get("frontier") != "unverifiable")
            rate_for_learnability = (
                stats.get("verified_pass_rate") if evidence_frontier else pass_rate
            ) if certified else None
            record["verified_pass_rate"] = stats.get("verified_pass_rate")
            record["learnability_basis"] = (
                "unverifiable" if not certified else
                "verified_visual" if evidence_frontier else "spy"
            )
        else:
            rate_for_learnability = pass_rate
        record["learnability"] = learnability(rate_for_learnability) if stats else 0.0
        record["reached_solver"] = bool(stats)
        scored.append(record)
    return scored


def build_proposer_sft_examples(
    scored_proposals: Sequence[Dict[str, Any]],
    threshold: float = DEFAULT_LEARNABILITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Keep the proposals that landed the solver in the middle, as SFT targets.

    This is the proposing side's training signal: the model is taught to repeat
    the proposals that turned out learnable, and nothing else. Same shape as the
    solver's SFT replay records so both train through one path.
    """
    examples: List[Dict[str, Any]] = []
    for proposal in scored_proposals:
        if (proposal.get("reached_solver") is False
                or proposal.get("learnability_basis") == "unverifiable"):
            continue
        if float(proposal.get("learnability", 0.0)) < threshold:
            continue
        if not proposal.get("completion") or not proposal.get("prompt"):
            continue
        example = {
            "task_id": proposal.get("proposal_id"),
            "problem": proposal.get("prompt"),
            "prompt": proposal.get("prompt"),
            "completion": proposal.get("completion"),
            "solution": proposal.get("completion"),
            "image": proposal.get("image_path"),
            "image_path": proposal.get("image_path"),
            "source_buffer": "proposer",
            "role": "proposer",
            "scene_id": proposal.get("scene_id"),
            "keep": proposal.get("keep"),
            "pass_rate": proposal.get("pass_rate"),
            "learnability": proposal.get("learnability"),
        }
        if "learnability_basis" in proposal:
            example["learnability_basis"] = proposal["learnability_basis"]
        if "verified_pass_rate" in proposal:
            example["verified_pass_rate"] = proposal["verified_pass_rate"]
        examples.append(example)
    return examples


def proposal_report(scored_proposals: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Round-level numbers for the proposing side; collapse shows up here first."""
    total = len(scored_proposals)
    if not total:
        return {"num_proposals": 0}
    valid = [p for p in scored_proposals if p.get("keep")]
    reached = [p for p in scored_proposals if p.get("reached_solver")]
    learn = [float(p.get("learnability", 0.0)) for p in reached]
    subsets = [tuple(p.get("keep") or ()) for p in valid]
    distinct = len(set(subsets))
    sizes: Dict[str, int] = {}
    for s in subsets:
        sizes[str(len(s))] = sizes.get(str(len(s)), 0) + 1
    return {
        "num_proposals": total,
        "num_parsed": len(valid),
        "parse_rate": round(len(valid) / total, 4),
        "num_reached_solver": len(reached),
        "mean_learnability": round(sum(learn) / len(learn), 4) if learn else None,
        "num_trainable": sum(
            1 for p in reached if p.get("solver_class") == "trainable"
        ),
        # A proposer that collapses onto one subset stops being an opponent;
        # this is the number that says so.
        "distinct_subsets": distinct,
        "subset_diversity": round(distinct / len(valid), 4) if valid else 0.0,
        "kept_size_histogram": dict(sorted(sizes.items())),
        "num_counterfactual_pairs": sum(
            1 for p in scored_proposals if p.get("pair_role") == "small"),
    }


def tag_counterfactual_pairs(
    proposals: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pair one-object edits with the same scene and requested player count.

    The counterfactual metric needs two tasks one object apart. Under self-play
    nobody plans pairs, but a scene's proposals often land next to each other
    anyway, so we pick those up for free instead of reserving slots for them.
    Each proposal joins at most one pair.
    """
    tagged = [dict(p) for p in proposals]
    for p in tagged:
        # Recomputing tags must not preserve a stale, potentially invalid pair.
        p.pop("pair_id", None)
        p.pop("pair_role", None)
    by_scene_and_players: Dict[tuple[str, Optional[int]], List[int]] = {}
    keep_sets: Dict[int, set[int]] = {}
    for i, p in enumerate(tagged):
        if not p.get("scene_id") or not isinstance(p.get("keep"), (list, tuple)):
            continue
        try:
            keep = [int(index) for index in p["keep"]]
            players = p.get("num_players")
            if players is not None:
                players = int(players)
        except (TypeError, ValueError):
            continue
        if (not keep or min(keep) < 0 or len(set(keep)) != len(keep)
                or (players is not None and players < 1)):
            continue
        keep_sets[i] = set(keep)
        # Missing counts are grouped only with other missing counts; the
        # generator gives both the same configured default player count.
        by_scene_and_players.setdefault(
            (str(p["scene_id"]), players), []
        ).append(i)

    for (scene_id, players), indices in by_scene_and_players.items():
        used: set = set()
        for a in indices:
            if a in used:
                continue
            small = keep_sets[a]
            for b in indices:
                if b == a or b in used or a in used:
                    continue
                large = keep_sets[b]
                if len(large) != len(small) + 1 or not small < large:
                    continue
                player_tag = "default" if players is None else str(players)
                pair_id = (f"{scene_id}::players{player_tag}::"
                           f"{'-'.join(map(str, sorted(small)))}|"
                           f"{'-'.join(map(str, sorted(large)))}")
                tagged[a]["pair_id"] = pair_id
                tagged[a]["pair_role"] = "small"
                tagged[b]["pair_id"] = pair_id
                tagged[b]["pair_role"] = "large"
                used.add(a)
                used.add(b)
                break
    return tagged
