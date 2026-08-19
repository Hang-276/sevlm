"""Policy-controlled, verifiable CLEVR task generator for the closed loop.

This replaces the old "shuffle the task pool and tag a focus" *replay* path as
the main generation route. It samples real CLEVR replacement scenes
from the image pool and emits candidate tasks whose ground truth and visual
evidence are derived from the scene metadata, so every task is verifiable.

Generation is controlled by ``generator_policy.json``:
  - ``image_selection_policy.seed_offset`` shifts which scenes are sampled, so
    iteration N+1 does not reuse iteration N's exact images;
  - ``difficulty_policy.distribution`` sets the target mix of easy/medium/hard;
  - ``sampling_policy.focus_weights`` biases generation toward failure modes.

Each candidate task carries the full field set the mentor requires:
    task_id, image_id, image_path, scene_id, scene_path, question/prompt,
    answer (ground_truth), rule_type, difficulty_target, difficulty_estimate,
    generation_reason, source_failure_profile, generator_policy_version.

No API calls, no model loading, no training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_r1.self_evolve.task_difficulty import (
    balance_by_label,
    estimate_difficulty,
    label_difficulty_correlation,
)
from open_r1.self_evolve.task_editor import (
    build_variant,
    counterfactual_pairs,
    enumerate_variants,
)

# Evidence before the answer, and symbolic coordinates — a concrete example box
# is the one thing an ungrounded model can profitably copy.
SPY_PLAYER_PROMPT = (
    "There are {num_players} players. Each player is associated with one image. One player "
    "is the spy because that player's image is different from the others. "
    "Identify the spy player and count the total number of changed attributes "
    "(color, shape, size, or material) across all changed objects.\n"
    "You MUST respond in EXACTLY this format:\n"
    "<think> your step-by-step reasoning </think>\n"
    "<bbox player=\"P\">[x1,y1,x2,y2]</bbox>\n"
    "<answer>spy=the player number; changed_attributes=the total count</answer>\n"
    "Output one <bbox> tag for every changed object, before the <answer> tag. "
    "In every <bbox> tag, set player to the spy player number (1 to "
    "{num_players}) and [x1,y1,x2,y2] to the pixel coordinates of the box "
    "around one object that differs in the spy's image, where the image is "
    "{image_width} pixels wide and {image_height} pixels high, with x1<x2 and "
    "y1<y2. At least one <bbox> tag is REQUIRED."
)

_ATTRS = ("color", "shape", "size", "material")


def _gold_boxes_from_modification(
    modification: Dict[str, Any],
    box_radius: int = 48,
    image_width: int = 320,
    image_height: int = 240,
) -> List[List[float]]:
    """Gold evidence pixel xyxy boxes from a scene's replaced objects.

    The replacement object's pixel_coords centre +/- radius, clamped to the
    image, so the candidate task carries the same gold boxes the grounding
    reward scores against.
    """
    boxes: List[List[float]] = []
    for ro in modification.get("replaced_objects", []) or []:
        repl = ro.get("replacement", {}) if isinstance(ro, dict) else {}
        pc = repl.get("pixel_coords")
        if pc and len(pc) >= 2:
            cx, cy = float(pc[0]), float(pc[1])
            boxes.append([
                max(0.0, cx - box_radius), max(0.0, cy - box_radius),
                min(float(image_width), cx + box_radius),
                min(float(image_height), cy + box_radius),
            ])
    return boxes


def _changed_objects_summary(modification: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact changed-object metadata (original→replacement attr diffs)."""
    out: List[Dict[str, Any]] = []
    for ro in modification.get("replaced_objects", []) or []:
        if not isinstance(ro, dict):
            continue
        orig = ro.get("original", {})
        repl = ro.get("replacement", {})
        changed = {a: [orig.get(a), repl.get(a)] for a in _ATTRS
                   if orig.get(a) != repl.get(a)}
        out.append({
            "index": ro.get("index"),
            "changed_attributes": changed,
            "replacement_pixel_coords": repl.get("pixel_coords"),
        })
    return out



@dataclass
class PolicyCLEVRGeneratorConfig:
    dataset_root: str
    iteration_id: str = "iter_000"
    num_tasks: int = 8
    seed: int = 42
    num_players: int = 5
    image_width: int = 320
    image_height: int = 240
    # "uniform" flattens the gold changed_attributes histogram across the
    # generated tasks; "none" restores the old pool-order behaviour.
    label_balance: str = "uniform"
    # Fraction of each round built by editing the previous round's seed scenes
    # instead of sampling fresh ones. 0 disables editing.
    edit_fraction: float = 0.5
    # Emit variants in counterfactual pairs (differ by exactly one object).
    counterfactual_pairs: bool = True
    variant_cache_dir: Optional[str] = None


def _histogram(values: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


class PolicyControlledCLEVRGenerator:
    """Generate verifiable CLEVR spy-player tasks under a generator policy."""

    def __init__(self, config: PolicyCLEVRGeneratorConfig):
        self.config = config
        self.root = Path(config.dataset_root)
        self.images_dir = self.root / "output" / "replacement_images"
        self.scenes_dir = self.root / "output" / "replacement_scenes"
        # Compose the REAL CLEVR spot-diff generator for scene-diff assembly
        # (spy sees modified image, civilians see original). The policy layer
        # here controls WHICH scene + difficulty + spy slot; the spot-diff
        # assembly itself is delegated to CLEVRSpotDiffGenerator so candidates
        # come from the genuine spot-diff path, not template replay.
        self._spotdiff = None
        self._generation_source = "clevr_spot_diff_generator"
        self._pool_estimates: Dict[str, Dict[str, Any]] = {}
        self.last_generation_report: Dict[str, Any] = {}
        try:
            from open_r1.clevr_spotdiff_generator import CLEVRSpotDiffGenerator

            self._spotdiff = CLEVRSpotDiffGenerator(
                images_dir=str(self.images_dir),
                scenes_dir=str(self.scenes_dir),
                num_players=config.num_players,
            )
        except Exception:
            # Adapter fallback: scene-diff metadata is still read directly from
            # replacement_scenes (NOT replay of old tasks); flagged accordingly.
            self._spotdiff = None
            self._generation_source = "clevr_spot_diff_adapter"

    # -- image pool ---------------------------------------------------------

    def _discover_base_names(self) -> List[str]:
        """Discover scenes that have original + modified images + scene json."""
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"CLEVR images dir not found: {self.images_dir}")
        base_names: List[str] = []
        for orig in sorted(self.images_dir.glob("*_original.png")):
            bn = orig.name[: -len("_original.png")]
            mod = self.images_dir / f"{bn}_modified.png"
            scene = self.scenes_dir / f"{bn}_comparison.json"
            if mod.is_file() and scene.is_file():
                base_names.append(bn)
        return base_names

    def _load_scene(self, base_name: str) -> Optional[Dict[str, Any]]:
        scene_path = self.scenes_dir / f"{base_name}_comparison.json"
        try:
            return json.loads(scene_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # -- difficulty-policy bucketing ---------------------------------------

    def _scene_estimate(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Difficulty of SEEING the change — see ``task_difficulty``.

        Never reads the changed-attribute count: that count is the gold answer,
        so bucketing on it turns the curriculum into a label prior.
        """
        objects = (scene.get("original_scene") or scene.get("scene") or {}).get("objects")
        return estimate_difficulty(
            scene.get("modification", {}),
            num_scene_objects=len(objects) if isinstance(objects, list) else None,
        )

    def _bucket_by_difficulty(
        self, base_names: List[str]
    ) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
        self._pool_estimates = {}
        for bn in base_names:
            scene = self._load_scene(bn)
            if not scene:
                continue
            est = self._scene_estimate(scene)
            self._pool_estimates[bn] = est
            buckets[est["difficulty_estimate"]].append(bn)
        return buckets

    def _label_leak_report(self, base_names: List[str]) -> Dict[str, Any]:
        """Difficulty/label correlation over the pool; |r| near 0 is the contract."""
        est = getattr(self, "_pool_estimates", {})
        rows = [est[bn] for bn in base_names if bn in est]
        r = label_difficulty_correlation(
            [row["difficulty_score"] for row in rows],
            [float(row["num_attr_changes"]) for row in rows],
        )
        histogram: Dict[str, int] = {}
        for row in rows:
            key = str(row["num_attr_changes"])
            histogram[key] = histogram.get(key, 0) + 1
        return {
            "difficulty_label_correlation": r,
            "pool_label_histogram": histogram,
            "pool_size": len(rows),
        }

    @staticmethod
    def _allocation(num_tasks: int, distribution: Dict[str, float]) -> Dict[str, int]:
        alloc = {k: int(round(num_tasks * float(distribution.get(k, 0.0))))
                 for k in ("easy", "medium", "hard")}
        # Fix rounding drift to exactly num_tasks.
        drift = num_tasks - sum(alloc.values())
        order = sorted(("easy", "medium", "hard"),
                       key=lambda k: -float(distribution.get(k, 0.0)))
        i = 0
        while drift != 0 and order:
            k = order[i % len(order)]
            if drift > 0:
                alloc[k] += 1
                drift -= 1
            elif alloc[k] > 0:
                alloc[k] -= 1
                drift += 1
            i += 1
        return alloc

    # -- editing ------------------------------------------------------------

    @property
    def _cache_dir(self) -> Path:
        return Path(self.config.variant_cache_dir
                    or (self.root / "output" / "variant_cache"))

    def _plan_edits(
        self,
        editing_policy: Dict[str, Any],
        rng: random.Random,
        exclude_scene_ids: Optional[set] = None,
        proposals: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Pick which subset-variants to build from last round's seed scenes.

        With self-play on, ``proposals`` carries the model's own picks and they
        take the edit slots. Otherwise ``direction`` comes from the seed's pass
        rate. Pairs are planned atomically (never split by the budget) and both
        members share one spy slot, so a pair differs in exactly one object and
        nothing else.
        """
        budget = int(self.config.num_tasks * float(self.config.edit_fraction))
        if budget <= 0:
            return []
        if proposals:
            return self._plan_from_proposals(proposals, rng, budget, exclude_scene_ids)
        seeds = list(editing_policy.get("seeds") or [])
        if not seeds:
            return []

        exclude = set(exclude_scene_ids or ())
        per_seed = 2 if self.config.counterfactual_pairs else 1
        plans: List[Dict[str, Any]] = []
        seen_scenes: set = set()
        for seed in seeds:
            if budget - len(plans) < per_seed:
                break
            scene_id = seed.get("scene_id")
            if not scene_id or scene_id in exclude or scene_id in seen_scenes:
                continue
            scene = self._load_scene(str(scene_id))
            if not scene:
                continue
            subsets = enumerate_variants(scene)
            pairs = counterfactual_pairs(subsets)
            if not pairs:
                continue
            seen_scenes.add(scene_id)
            small, large = pairs[rng.randrange(len(pairs))]
            pair_id = f"{scene_id}::{'-'.join(map(str, small))}|{'-'.join(map(str, large))}"
            spy_player = rng.randrange(self.config.num_players) + 1
            harder = seed.get("direction") == "harder"
            primary, other = (large, small) if harder else (small, large)
            plans.append({"scene_id": scene_id, "keep": primary, "pair_id": pair_id,
                          "pair_role": "large" if primary is large else "small",
                          "spy_player": spy_player, "seed_regret": seed.get("regret")})
            if self.config.counterfactual_pairs:
                plans.append({"scene_id": scene_id, "keep": other, "pair_id": pair_id,
                              "pair_role": "small" if primary is large else "large",
                              "spy_player": spy_player, "seed_regret": seed.get("regret")})
        return plans

    def _plan_from_proposals(
        self,
        proposals: List[Dict[str, Any]],
        rng: random.Random,
        budget: int,
        exclude_scene_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Take the model's own picks as the edit plan, one slot per proposal.

        Each proposal is its own task, so a scene's group of proposals becomes
        that scene's group of tasks — the solver budget is unchanged, we only
        moved who chooses. Proposals from one scene share a spy slot so they
        stay comparable with each other.
        """
        exclude = set(exclude_scene_ids or ())
        spy_by_scene: Dict[str, int] = {}
        plans: List[Dict[str, Any]] = []
        # A variant's task_id is derived from (scene, kept subset), so two
        # proposals that picked the same subset would collide into one task_id:
        # their rollouts would merge into a single oversized group and the pass
        # rate would be computed over the wrong set. The action space is small,
        # so duplicate picks are common — keep the first, drop the rest.
        seen_subsets: set = set()
        for proposal in proposals:
            if len(plans) >= budget:
                break
            scene_id = proposal.get("scene_id")
            keep = proposal.get("keep")
            if not scene_id or not keep or scene_id in exclude:
                continue
            scene_id = str(scene_id)
            subset_key = (scene_id, tuple(sorted(keep)))
            if subset_key in seen_subsets:
                continue
            seen_subsets.add(subset_key)
            players = int(proposal.get("num_players") or self.config.num_players)
            if scene_id not in spy_by_scene:
                # The spy slot is drawn here, never taken from the proposal: it
                # IS the answer, so a proposer that could pick it would let both
                # roles settle on one constant slot and call it a win.
                spy_by_scene[scene_id] = rng.randrange(players) + 1
            if spy_by_scene[scene_id] > players:
                spy_by_scene[scene_id] = rng.randrange(players) + 1
            plans.append({
                "scene_id": scene_id,
                "keep": list(keep),
                "num_players": proposal.get("num_players"),
                "spy_player": spy_by_scene[scene_id],
                "proposal_id": proposal.get("proposal_id"),
                "pair_id": proposal.get("pair_id"),
                "pair_role": proposal.get("pair_role"),
                "source": "proposer",
            })
        return plans

    def _build_variant_tasks(
        self, edit_plan: List[Dict[str, Any]], prompt: str, policy_version: int
    ) -> List[Dict[str, Any]]:
        """Build all planned variants; a planned pair whose member fails is dropped whole.

        Proposals are exempt: nobody asked for those pairs, they were noticed
        after the fact, so a buildable proposal is kept even when the sibling it
        happened to pair with failed.
        """
        from_proposer = any(p.get("source") == "proposer" for p in edit_plan)
        built: List[Dict[str, Any]] = []
        by_pair: Dict[str, List[Dict[str, Any]]] = {}
        for i, plan in enumerate(edit_plan):
            task = self._build_variant_task(plan, prompt, policy_version, i)
            if task is None:
                by_pair.setdefault(plan.get("pair_id") or f"solo{i}", []).append(None)
            else:
                by_pair.setdefault(task.get("pair_id") or f"solo{i}", []).append(task)
        paired = self.config.counterfactual_pairs and not from_proposer
        for pair_id, members in by_pair.items():
            ok = [m for m in members if m is not None]
            if paired and pair_id and not str(pair_id).startswith("solo"):
                if len(ok) == len(members):
                    built.extend(ok)   # complete pair
                # else: drop the orphan — a half pair is neither a contrast nor
                # worth its slot; fresh sampling refills it.
            else:
                built.extend(ok)
        return built

    def _build_variant_task(
        self, plan: Dict[str, Any], prompt: str, policy_version: int, index: int
    ) -> Optional[Dict[str, Any]]:
        """Turn one edit plan into a candidate task with an exact gold answer."""
        scene_id = str(plan["scene_id"])
        from_proposer = plan.get("source") == "proposer"
        source = "self_play_variant" if from_proposer else "regret_seeded_variant"
        scene = self._load_scene(scene_id)
        if not scene:
            return None
        variant = build_variant(scene_id, scene, plan["keep"], self.images_dir, self._cache_dir)
        if not variant:
            return None

        num_players = int(plan.get("num_players") or self.config.num_players)
        spy_player = int(plan.get("spy_player")
                         or ((index + policy_version) % num_players) + 1)
        # A proposed task can ask for its own number of players, so its prompt
        # has to state that number rather than the generator-wide default.
        if num_players != self.config.num_players:
            prompt = SPY_PLAYER_PROMPT.format(
                num_players=num_players,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        image_path = [
            variant["spy_image_path"] if (p + 1) == spy_player else variant["civilian_image_path"]
            for p in range(num_players)
        ]
        gold_answer = f"spy={spy_player}; changed_attributes={variant['num_attr_changes']}"
        # Difficulty of the VARIANT: only the kept objects are still changed.
        replaced = scene.get("modification", {}).get("replaced_objects") or []
        kept_mod = {"replaced_objects": [replaced[i] for i in plan["keep"]]}
        objects = (scene.get("original_scene") or scene.get("scene") or {}).get("objects")
        est = estimate_difficulty(
            kept_mod, num_scene_objects=len(objects) if isinstance(objects, list) else None
        )
        task_id = f"{self.config.iteration_id}::{variant['variant_id']}"
        return {
            "task_id": task_id,
            "base_task_id": variant["variant_id"],
            "image_id": variant["variant_id"],
            "image_path": image_path,
            "scene_id": scene_id,
            "scene_path": str(self.scenes_dir / f"{scene_id}_comparison.json"),
            "problem": prompt,
            "prompt": prompt,
            "answer": f"<answer>{gold_answer}</answer>",
            "solution": f"<answer>{gold_answer}</answer>",
            "ground_truth": gold_answer,
            "rule_type": "clevr_spy_change_count",
            "difficulty_target": est["difficulty_estimate"],
            "difficulty_estimate": est["difficulty_estimate"],
            "generation_reason": (
                f"proposed from {scene_id} (keep={plan['keep']})"
                if from_proposer else
                f"edited from {scene_id} (keep={plan['keep']}, "
                f"seed_regret={plan.get('seed_regret')})"
            ),
            "generator_policy_version": policy_version,
            "generation_source": source,
            "proposal_id": plan.get("proposal_id"),
            "pair_id": plan.get("pair_id"),
            "pair_role": plan.get("pair_role"),
            "gold_bbox": variant["gold_boxes"],
            "metadata": {
                "base_name": scene_id,
                "num_players": num_players,
                "spy_player": spy_player,
                "proposal_id": plan.get("proposal_id"),
                "gold_evidence_boxes": variant["gold_boxes"],
                "variant": variant,
                "pair_id": plan.get("pair_id"),
                "pair_role": plan.get("pair_role"),
                "difficulty_metadata": est,
            },
            "generator": {
                "iteration_id": self.config.iteration_id,
                "source": source,
                "generator_policy_version": policy_version,
                "rule_type": "clevr_spy_change_count",
            },
            "evidence_schema": "clevr_scene_metadata_replaced_objects",
        }

    # -- main generation ----------------------------------------------------

    def generate(
        self,
        generator_policy: Dict[str, Any],
        failure_profile: Optional[Dict[str, Any]] = None,
        exclude_scene_ids: Optional[set] = None,
        proposals: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ``num_tasks`` verifiable candidate tasks under the policy.

        ``exclude_scene_ids`` lets the regenerate path avoid re-proposing scenes
        that were already rejected this iteration. ``proposals`` carries the
        model's own subset picks when self-play is on; they take the edit slots
        and the rest of the round is still sampled fresh.
        """
        policy = generator_policy or {}
        policy_version = int(policy.get("generator_policy_version", 0))
        difficulty_policy = policy.get("difficulty_policy", {}) or {}
        distribution = difficulty_policy.get("distribution", {"easy": 0.3, "medium": 0.5, "hard": 0.2})
        difficulty_target = difficulty_policy.get("difficulty_target", "medium")
        image_policy = policy.get("image_selection_policy", {}) or {}
        seed_offset = int(image_policy.get("seed_offset", 0))
        focus_weights = (policy.get("sampling_policy", {}) or {}).get("focus_weights", {})

        editing_policy = policy.get("editing_policy", {}) or {}
        exclude_scene_ids = set(exclude_scene_ids or set())
        # Mastered scenes carry no headroom; stop proposing them.
        exclude_scene_ids |= {s for s in editing_policy.get("retire_scene_ids", []) if s}

        base_names = self._discover_base_names()
        if not base_names:
            raise ValueError("CLEVR image pool is empty — cannot generate tasks.")

        # The seed offset (advanced each iteration by the policy update) shifts
        # the sampling so iteration N+1 picks a different image window than N.
        rng = random.Random(self.config.seed + 1000 * seed_offset)

        buckets = self._bucket_by_difficulty(base_names)
        for lvl in buckets:
            buckets[lvl] = [b for b in buckets[lvl] if b not in exclude_scene_ids]
            rng.shuffle(buckets[lvl])

        # Edited variants of last round's seed scenes take the first slots;
        # the rest of the round is sampled fresh.
        prompt = SPY_PLAYER_PROMPT.format(
            num_players=self.config.num_players,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        edit_plan = self._plan_edits(editing_policy, rng, exclude_scene_ids, proposals)
        variant_tasks = self._build_variant_tasks(edit_plan, prompt, policy_version)
        # Fresh sampling fills whatever the (possibly failed) edits left open,
        # and never re-proposes a scene the edits already used.
        exclude_scene_ids |= {t["scene_id"] for t in variant_tasks}
        num_fresh = max(0, self.config.num_tasks - len(variant_tasks))
        alloc = self._allocation(num_fresh, distribution)

        # Difficulty decides how hard the task is; label balancing decides what
        # the answer is. Keeping them independent is the point of the redesign.
        label_of = lambda bn: self._pool_estimates.get(bn, {}).get("num_attr_changes")
        chosen: List[tuple] = []  # (base_name, difficulty_estimate)
        for lvl in ("easy", "medium", "hard"):
            want = alloc.get(lvl, 0)
            take = balance_by_label(buckets[lvl], label_of, want, self.config.label_balance)
            chosen += [(bn, lvl) for bn in take]
            buckets[lvl] = [b for b in buckets[lvl] if b not in set(take)]

        # Backfill if some buckets were short (keeps the round at num_tasks).
        if len(chosen) < num_fresh:
            leftover = [bn for lvl in ("medium", "easy", "hard") for bn in buckets[lvl]]
            rng.shuffle(leftover)
            fill = balance_by_label(
                leftover, label_of, num_fresh - len(chosen), self.config.label_balance,
            )
            chosen += [(bn, self._pool_estimates.get(bn, {}).get("difficulty_estimate", "medium"))
                       for bn in fill]

        self.last_generation_report = {
            **self._label_leak_report(base_names),
            "selected_label_histogram": _histogram(
                [label_of(bn) for bn, _ in chosen]
            ),
            "label_balance": self.config.label_balance,
            "difficulty_allocation": alloc,
            "num_edited": len(variant_tasks),
            "num_fresh": len(chosen),
            "num_counterfactual_pairs": sum(
                1 for t in variant_tasks if t.get("pair_role") == "small"),
            "edit_source": "proposer" if proposals else "regret_seeded",
        }

        candidates: List[Dict[str, Any]] = list(variant_tasks)
        offset = len(candidates)
        for i, (bn, est_level) in enumerate(chosen, start=offset):
            scene = self._load_scene(bn)
            if not scene:
                continue
            modification = scene.get("modification", {})
            est = self._scene_estimate(scene)

            # Deterministic spy placement (varies with policy version so the
            # answer distribution shifts across iterations too).
            spy_player = ((i + policy_version) % self.config.num_players) + 1

            # Delegate spot-diff image assembly to the REAL generator when
            # available (spy↔modified, civilians↔original). Fall back to direct
            # scene-metadata assembly (adapter) otherwise.
            generation_source = self._generation_source
            if self._spotdiff is not None:
                game = self._spotdiff.build_spy_game_for_base(bn, spy_player, i)
                if game and game.get("player_images"):
                    image_path = list(game["player_images"])
                    spy_player = int(game["spy_player"])
                else:
                    generation_source = "clevr_spot_diff_adapter"
                    orig = str(self.images_dir / f"{bn}_original.png")
                    mod = str(self.images_dir / f"{bn}_modified.png")
                    image_path = [
                        mod if (p + 1) == spy_player else orig
                        for p in range(self.config.num_players)
                    ]
            else:
                orig = str(self.images_dir / f"{bn}_original.png")
                mod = str(self.images_dir / f"{bn}_modified.png")
                image_path = [
                    mod if (p + 1) == spy_player else orig
                    for p in range(self.config.num_players)
                ]

            gold_boxes = _gold_boxes_from_modification(
                modification,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
            changed_objects = _changed_objects_summary(modification)
            gold_answer = (
                f"spy={spy_player}; changed_attributes={est['num_attr_changes']}"
            )

            top_focus = max(focus_weights, key=focus_weights.get) if focus_weights else None
            generation_reason = (
                f"{generation_source} | policy v{policy_version}: "
                f"difficulty_target={difficulty_target}, sampled_bucket={est_level}"
                + (f", focus={top_focus}" if top_focus else "")
            )

            candidates.append({
                "task_id": f"{self.config.iteration_id}::{bn}",
                "base_task_id": bn,
                "image_id": bn,
                "image_path": image_path,
                "scene_id": bn,
                "scene_path": str(self.scenes_dir / f"{bn}_comparison.json"),
                "problem": prompt,
                "prompt": prompt,
                "answer": f"<answer>{gold_answer}</answer>",
                "solution": f"<answer>{gold_answer}</answer>",
                "ground_truth": gold_answer,
                "rule_type": "clevr_spy_change_count",
                "difficulty_target": difficulty_target,
                "difficulty_estimate": est["difficulty_estimate"],
                "generation_reason": generation_reason,
                "source_failure_profile": {
                    "failure_tag_counts": (failure_profile or {}).get("failure_tag_counts", {}),
                    "focus_weights": focus_weights,
                },
                "generator_policy_version": policy_version,
                "generation_source": generation_source,
                "gold_bbox": gold_boxes,
                "changed_objects": changed_objects,
                "metadata": {
                    "base_name": bn,
                    "num_players": self.config.num_players,
                    "spy_player": spy_player,
                    "gold_evidence_boxes": gold_boxes,
                    "comparison_data": {
                        "replaced_objects": modification.get("replaced_objects", []),
                        "gold_evidence_boxes": gold_boxes,
                        "changed_objects": changed_objects,
                    },
                    "difficulty_metadata": est,
                },
                "generator": {
                    "iteration_id": self.config.iteration_id,
                    "source": generation_source,
                    "generation_source": generation_source,
                    "generator_policy_version": policy_version,
                    "difficulty_target": difficulty_target,
                    "rule_type": "clevr_spy_change_count",
                    "delegated_to": "CLEVRSpotDiffGenerator"
                    if self._spotdiff is not None else "scene_metadata_adapter",
                },
                "evidence_schema": "clevr_scene_metadata_replaced_objects",
            })
            if len(candidates) >= self.config.num_tasks:
                break

        return candidates
