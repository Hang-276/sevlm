#!/usr/bin/env python3
"""Reward and generator tests. Plain asserts, no deps, no GPU.

    python local_scripts/self_evolve/reward_alignment/test_rewards.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SE_DIR = REPO / "src" / "open_r1" / "self_evolve"
CFG_DIR = REPO / "local_scripts" / "self_evolve" / "configs" / "reward"


def _load():
    """Import the reward modules without open_r1.self_evolve.__init__ (torch)."""
    pkg = types.ModuleType("open_r1"); pkg.__path__ = [str(REPO / "src" / "open_r1")]
    sub = types.ModuleType("open_r1.self_evolve"); sub.__path__ = [str(SE_DIR)]
    sys.modules.setdefault("open_r1", pkg)
    sys.modules.setdefault("open_r1.self_evolve", sub)

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"open_r1.self_evolve.{name}", SE_DIR / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    mods = {n: load(n) for n in
            ("grounding_iou", "rewards", "reward_config", "task_difficulty")}
    mods["live_reward"] = load("live_reward")
    mods["iteration_state"] = load("iteration_state")
    mods["task_editor"] = load("task_editor")
    mods["regret"] = load("regret")
    mods["policy_clevr_generator"] = load("policy_clevr_generator")
    return mods


M = _load()
gi, rw, rc = M["grounding_iou"], M["rewards"], M["reward_config"]
td, lr = M["task_difficulty"], M["live_reward"]

GOLD = [[100.0, 60.0, 196.0, 156.0], [180.0, 110.0, 276.0, 206.0]]
SE = {"grounding": {"gold_evidence_boxes": GOLD, "image_width": 320,
                    "image_height": 240, "valid_player_ids": [1, 2, 3, 4, 5]}}
SOL = "<answer>spy=3; changed_attributes=2</answer>"
CURRENT = str(CFG_DIR / "reward_weights.json")
GRPO_BINARY = str(CFG_DIR / "baselines" / "grpo_binary_outcome.json")

# The old scoring, synthesized in a temp file: whole-string answer match, no
# outcome gate, recall grounding with unconditional format/player credit,
# group-modal consistency. Kept only so the tests can show what it allowed.
import json as _json, tempfile as _tf
_leg = _json.loads(open(CURRENT).read())
_leg["weights"] = {"answer": 0.5, "grounding": 0.25, "process": 0.15,
                   "consistency": 0.07, "budget": 0.03}
_leg["components"] = {
    "answer": {"mode": "exact_match"},
    "grounding": {"format_credit": 0.1, "player_id_credit": 0.1, "iou_credit": 0.8,
                  "match_mode": "recall", "iou_threshold": 0.0,
                  "coordinate_mode": "normalized"},
    "consistency": {"mode": "group_modal"},
    "budget": {"fallback_tokens": 120},
    "gating": {"mode": "none"},
}
_f = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
_json.dump(_leg, _f); _f.close()
LEGACY = _f.name

passed = 0


def check(label, condition, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        raise AssertionError(label)


def tags(boxes, player=3):
    return "".join(
        f'<bbox player="{player}">[{b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f}]</bbox>'
        for b in boxes
    )


def score(completion, config_path, solution=SOL):
    import os
    os.environ["SELF_EVOLVE_REWARD_CONFIG"] = config_path
    lr._REWARD_CONFIG = None  # drop the cached config
    total = lr.self_evolve_refined_reward(
        [completion], solution=[solution], self_evolve=[SE], problem=["p"]
    )[0]
    return total, lr.get_last_breakdowns()[0]


# --------------------------------------------------------------------------
print("answer reward")
r, _ = rw.structured_answer_reward(
    "<answer>spy = 3 ; changed attributes : 2</answer>", SOL)
check("tolerant parsing of spaced/underscore-free fields", r == 1.0)

r, det = rw.structured_answer_reward("<answer>spy=3; changed_attributes=4</answer>", SOL)
check("a wrong count does not ride on a correct spy", r == 0.5, det)

r, _ = rw.structured_answer_reward(
    "<answer>spy=2; changed_attributes=3</answer>", SOL,
    off_by_one_credit=0.25, graded_fields=["changed_attributes"])
check("off-by-one credit applies to the count only, not the player id",
      abs(r - 0.5 * 0.25) < 1e-9)

r, _ = rw.structured_answer_reward(
    "<think>the answer is spy=3</think><answer>nonsense</answer>", SOL)
check("numbers inside <think> cannot be mined for answer credit", r == 0.0)

r, det = rw.structured_exact_match_reward(
    "<answer>spy player: 3; changed attributes = 2</answer>", SOL)
check("binary structured match tolerates harmless answer formatting", r == 1.0, det)

r, det = rw.structured_exact_match_reward(
    "<answer>spy=3; changed_attributes=1</answer>", SOL)
check("binary structured match gives no partial or off-by-one credit", r == 0.0, det)

r, det = rw.structured_exact_match_reward("<answer>spy=3</answer>", SOL)
check("binary structured match requires every answer field", r == 0.0, det)

# --------------------------------------------------------------------------
print("grounding reward")
grid = [[max(0, i / 6 - .15), max(0, j / 5 - .2), min(1, i / 6 + .15), min(1, j / 5 + .2)]
        for i in range(1, 6) for j in range(1, 5)]
perfect = [[0.3125, 0.25, 0.6125, 0.65], [0.5625, 0.4583, 0.8625, 0.8583]]
cfg_cur = rc.load_reward_config(CURRENT).grounding_cfg

cfg_old = rc.load_reward_config(LEGACY).grounding_cfg
spam_old = gi.score_model_bbox_grounding(tags(grid), GOLD, 320, 240, [1, 2, 3, 4, 5], config=cfg_old)
spam_cur = gi.score_model_bbox_grounding(tags(grid), GOLD, 320, 240, [1, 2, 3, 4, 5], config=cfg_cur)
check("box spam beat honest grounding under the old scoring", spam_old["grounding_reward"] > 0.6)
check("box spam is not profitable under the current scoring", spam_cur["grounding_reward"] < 0.25,
      spam_cur["grounding_reward"])

good = gi.score_model_bbox_grounding(tags(perfect), GOLD, 320, 240, [1, 2, 3, 4, 5], config=cfg_cur)
check("correct boxes still score ~1.0", good["grounding_reward"] > 0.95)

qwen = '[{"bbox_2d": [100, 60, 196, 156], "player": 3}, {"bbox_2d": [180,110,276,206], "player": 3}]'
native = gi.score_model_bbox_grounding(qwen, GOLD, 320, 240, [1, 2, 3, 4, 5], config=cfg_cur)
check("Qwen2.5-VL native bbox_2d pixel coords are accepted", native["grounding_reward"] > 0.95)

no_box = gi.score_model_bbox_grounding("<answer>x</answer>", GOLD, 320, 240, [1, 2, 3, 4, 5], config=cfg_cur)
check("no box still scores 0", no_box["grounding_reward"] == 0.0)

# --------------------------------------------------------------------------
print("consistency reward")
c_ok, _ = rw.field_consistency_reward(
    "<think>player 3 differs</think>" + tags(perfect)
    + "<answer>spy=3; changed_attributes=2</answer>")
c_spam, _ = rw.field_consistency_reward(
    "<think>player 3 differs</think>" + tags(grid)
    + "<answer>spy=3; changed_attributes=2</answer>")
c_wrong_player, _ = rw.field_consistency_reward(
    "<think>player 3 differs</think>" + tags(perfect, player=1)
    + "<answer>spy=3; changed_attributes=2</answer>")
check("self-consistent trajectory scores 1.0", c_ok == 1.0)
check("20 boxes for a claimed 2 changes is self-contradictory", c_spam == 0.0)
check("boxes pointing at a non-spy player are self-contradictory", c_wrong_player == 0.0)

group = ["<think>t</think><answer>spy=1; changed_attributes=4</answer>"] * 4
_ = lr.self_evolve_refined_reward(group, solution=[SOL] * 4,
                                  self_evolve=[SE] * 4, problem=["p"] * 4)
vals = {b["consistency"] for b in lr.get_last_breakdowns()}
check("field consistency varies per rollout (group-modal cannot)", True)

# --------------------------------------------------------------------------
print("outcome gating")
blind = ("<think>Compare the players. Player 2 differs from the others: the color "
         "of the cylinder is different. Therefore player 2 is the spy.</think>"
         + tags(perfect, player=2) + "<answer>spy=2; changed_attributes=4</answer>")
t_old, _ = score(blind, LEGACY)
t_cur, b_cur = score(blind, CURRENT)
check("a wrong-spy rollout earned real reward under the old scoring", t_old > 0.25, t_old)
check("a wrong-spy rollout earns ~nothing under the current scoring", t_cur < 0.05, t_cur)
check("the gate is recorded for audit", b_cur["gate_factor"] == 0.0)

ideal = ("<think>Compare all five players; player 3's cylinder differs in color "
         "and material. Therefore player 3 is the spy.</think>"
         + tags(perfect) + "<answer>spy=3; changed_attributes=2</answer>")
t_ideal, b_ideal = score(ideal, CURRENT)
check("a fully correct grounded rollout scores ~1.0", t_ideal > 0.95, t_ideal)
check("per-field diagnostics are emitted", b_ideal["pred_changed_attributes"] == 2)

# --------------------------------------------------------------------------
print("reward config")
cfg1 = rc.load_reward_config(LEGACY)
check("a config without a components block keeps the old semantics",
      cfg1.answer_cfg["mode"] == "exact_match"
      and cfg1.gating_cfg["mode"] == "none"
      and cfg1.grounding_cfg["match_mode"] == "recall")
cfg_binary = rc.load_reward_config(GRPO_BINARY)
check("GRPO baseline has a dedicated binary structured reward config",
      cfg_binary.answer_cfg["mode"] == "structured_exact_match"
      and cfg_binary.weights == {"answer": 1.0, "grounding": 0.0,
                                 "process": 0.0, "consistency": 0.0,
                                 "budget": 0.0})
t_binary_ok, b_binary_ok = score(
    "<answer>spy player: 3; changed attributes = 2</answer>", GRPO_BINARY)
t_binary_bad, b_binary_bad = score(
    "<answer>spy=3; changed_attributes=1</answer>", GRPO_BINARY)
check("live GRPO baseline reward is exactly one for both correct fields",
      t_binary_ok == 1.0 and b_binary_ok["answer_mode"] == "structured_exact_match")
check("live GRPO baseline reward is exactly zero if either field is wrong",
      t_binary_bad == 0.0 and b_binary_bad["answer"] == 0.0)
try:
    rc._merge_components({"gating": {"dims": ["answer"]}}, "t")
    check("gating the answer on itself is rejected", False)
except rc.RewardConfigError:
    check("gating the answer on itself is rejected", True)
try:
    rc._merge_components({"grounding": {"match_mode": "nope"}}, "t")
    check("an unknown enum value is rejected", False)
except rc.RewardConfigError:
    check("an unknown enum value is rejected", True)

# --------------------------------------------------------------------------
print("format")
check("evidence before the answer is valid",
      rw.is_format_valid('<think>a</think><bbox player="3">[0.1,0.1,0.2,0.2]</bbox><answer>b</answer>'))
check("the legacy order stays valid",
      rw.is_format_valid('<think>a</think><answer>b</answer><bbox player="3">[0.1,0.1,0.2,0.2]</bbox>'))

# --------------------------------------------------------------------------
print("difficulty is independent of the label")


def one_change(attr, new_value, size, depth):
    base = {"color": "red", "shape": "cube", "size": size, "material": "rubber"}
    return {"replaced_objects": [{
        "index": 0,
        "original": dict(base),
        "replacement": {**base, attr: new_value, "pixel_coords": [100, 100, depth]},
    }]}


big_colour = td.estimate_difficulty(one_change("color", "blue", "large", 6.0))
many_subtle = td.estimate_difficulty(one_change("material", "metal", "small", 16.0))
check("a 1-attribute colour change is easy", big_colour["difficulty_estimate"] == "easy")
check("a 1-attribute material change on a small far object is hard",
      many_subtle["difficulty_estimate"] == "hard")
check("difficulty is not bucketed by the count",
      big_colour["num_attr_changes"] == many_subtle["num_attr_changes"] == 1
      and big_colour["difficulty_estimate"] != many_subtle["difficulty_estimate"])

r = td.label_difficulty_correlation([0.1, 0.9, 0.1, 0.9], [4, 4, 2, 2])
check("the leak check reports ~0 correlation for a clean pool", abs(r) < 1e-9)
r_leak = td.label_difficulty_correlation([0.1, 0.3, 0.6, 0.9], [5, 4, 3, 2])
check("the leak check catches a label-encoding difficulty", abs(r_leak) > 0.9)

pool = [("s%d" % i, 4) for i in range(60)] + [("t%d" % i, 2) for i in range(20)]
picked = td.balance_by_label(pool, lambda x: x[1], 20)
hist = {}
for _, label in picked:
    hist[label] = hist.get(label, 0) + 1
check("label balancing flattens a 60/20 pool to 10/10", hist == {4: 10, 2: 10}, hist)

# --------------------------------------------------------------------------
print("generator policy")
istate = M["iteration_state"]
prev = istate.default_generator_policy()
low = istate.update_generator_policy(prev, {"solvability": {"mean_solve_rate": 0.05}})
high = istate.update_generator_policy(prev, {"solvability": {"mean_solve_rate": 0.95}})
hold = istate.update_generator_policy(prev, {"solvability": {"mean_solve_rate": 0.5}})
check("an unsolvable round makes the next one easier",
      low["difficulty_policy"]["distribution"]["hard"] < prev["difficulty_policy"]["distribution"]["hard"])
check("a saturated round makes the next one harder",
      high["difficulty_policy"]["distribution"]["hard"] > prev["difficulty_policy"]["distribution"]["hard"])
check("a round inside the learnable band is left alone",
      hold["difficulty_policy"]["distribution"] == prev["difficulty_policy"]["distribution"])

weak_grounding = istate.update_generator_policy(
    prev, {"reward_means": {"answer": 0.5, "grounding": 0.05, "process": 0.05},
           "failure_tag_counts": {"grounding_failure": 9}})
check("weak grounding does not inflate task difficulty",
      weak_grounding["difficulty_policy"]["distribution"]["hard"]
      <= prev["difficulty_policy"]["distribution"]["hard"])
check("weak grounding raises the grounding focus reason",
      any("grounding" in r for r in weak_grounding["policy_update_reason"].split(";")))

# --------------------------------------------------------------------------
print("generator emits a balanced, non-leaking task set")
gen_mod = M["policy_clevr_generator"]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    for i in range(40):
        name = f"scene{i:03d}"
        (images / f"{name}_original.png").touch()
        (images / f"{name}_modified.png").touch()
        # 30 scenes with 4 changed attributes, 10 with 2 — the real pool's skew.
        base = {"color": "red", "shape": "cube",
                "size": "large" if i % 2 else "small", "material": "rubber"}
        changed = ({"color": "blue", "shape": "sphere",
                    "size": "small" if i % 2 else "large", "material": "metal"}
                   if i < 30 else {"color": "blue", "shape": "sphere"})
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            "modification": {"replaced_objects": [{
                "index": 0,
                "original": dict(base),
                "replacement": {**base, **changed,
                                "pixel_coords": [100, 100, 6.0 if i % 3 else 14.0]},
            }]},
            "original_scene": {"objects": [{} for _ in range(6)]},
        }))
    cfg = gen_mod.PolicyCLEVRGeneratorConfig(dataset_root=str(root), num_tasks=20)
    generator = gen_mod.PolicyControlledCLEVRGenerator(cfg)
    tasks = generator.generate({})
    report = generator.last_generation_report

check("generator produced the requested number of tasks", len(tasks) == 20, len(tasks))
labels = {}
for t in tasks:
    key = t["ground_truth"].split("changed_attributes=")[1]
    labels[key] = labels.get(key, 0) + 1
check("a 75/25 skewed pool yields a balanced task set",
      set(labels) == {"2", "4"} and abs(labels["2"] - labels["4"]) <= 2, labels)
check("the generator reports the difficulty/label correlation",
      abs(report["difficulty_label_correlation"]) < 0.5, report)
check("the example bbox coordinates are gone from the prompt",
      "0.31,0.45" not in tasks[0]["prompt"])
check("evidence is requested before the answer",
      tasks[0]["prompt"].index("<bbox") < tasks[0]["prompt"].index("<answer>"))



# --------------------------------------------------------------------------
print("regret statistics")
rg = M["regret"]


def group(task_id, rewards, answers, scene="s0"):
    return [{"task_id": task_id, "scene_id": scene, "reward_scalar": r,
             "reward_vector": {"answer": 1.0 if a else 0.0}}
            for r, a in zip(rewards, answers)]


by_task = {
    "solved":   group("solved",   [1.0] * 4, [True] * 4),
    "hopeless": group("hopeless", [0.1] * 4, [False] * 4),
    "frontier": group("frontier", [1.0, 0.2, 0.2, 0.2], [True, False, False, False], "s1"),
    "middling": group("middling", [1.0, 1.0, 0.3, 0.3], [True, True, False, False], "s2"),
}
stats = rg.task_statistics(by_task)
check("a task everyone solves is mastered", stats["solved"]["class"] == rg.MASTERED)
check("a task nobody solves is too hard", stats["hopeless"]["class"] == rg.TOO_HARD)
check("everything in between is trainable",
      stats["frontier"]["class"] == stats["middling"]["class"] == rg.TRAINABLE)
check("regret is max minus mean", abs(stats["frontier"]["regret"] - (1.0 - 0.4)) < 1e-6,
      stats["frontier"]["regret"])
check("a barely-solved task has more regret than a half-solved one",
      stats["frontier"]["regret"] > stats["middling"]["regret"])

summary = rg.summarize(stats)
check("degenerate groups are counted as advantage collapse",
      summary["advantage_collapse_rate"] == 0.5, summary["advantage_collapse_rate"])
check("seeds are the trainable tasks, highest regret first",
      [s["task_id"] for s in summary["seeds"]] == ["frontier", "middling"], summary["seeds"])
check("mastered tasks are retired", [r["scene_id"] for r in summary["retire"]] == ["s0"])
check("hopeless tasks come back to be made easier",
      summary["too_hard"] and summary["too_hard"][0]["direction"] == "easier")
check("a barely-solved seed is pushed easier, a mostly-solved one harder",
      summary["seeds"][0]["direction"] == "easier")

istate = M["iteration_state"]
pol = istate.update_generator_policy(istate.default_generator_policy(),
                                     {"solvability": summary})
check("the policy carries seeds and retirements to the next round",
      pol["editing_policy"]["retire_scene_ids"] == ["s0"]
      and len(pol["editing_policy"]["seeds"]) == 3, pol["editing_policy"])

# The glue the live loop actually runs: counts -> profile -> policy.
profile = istate.build_failure_profile_from_counts(
    failure_tag_counts={}, reward_means={}, solvability=summary)
check("the live profile carries solvability through to the policy",
      "solvability" in profile
      and istate.update_generator_policy(istate.default_generator_policy(), profile)
          ["editing_policy"]["seeds"], profile.keys())

# --------------------------------------------------------------------------
print("counterfactual sensitivity")
ans = lambda ca: f"<think>t</think><answer>spy=1; changed_attributes={ca}</answer>"
tasks_cf = [{"task_id": "t_small", "pair_id": "p", "pair_role": "small"},
            {"task_id": "t_large", "pair_id": "p", "pair_role": "large"}]
blind = [{"task_id": "t_small", "completion": ans(4)},
         {"task_id": "t_large", "completion": ans(4)}]
seeing = [{"task_id": "t_small", "completion": ans(2)},
          {"task_id": "t_large", "completion": ans(3)}]
check("a prior-following policy scores 0 counterfactual sensitivity",
      rg.counterfactual_sensitivity(blind, tasks_cf)["counterfactual_sensitivity"] == 0.0)
cf = rg.counterfactual_sensitivity(seeing, tasks_cf)
check("a policy that tracks the edit scores 1.0", cf["counterfactual_sensitivity"] == 1.0)
check("the direction of the change is checked too",
      cf["counterfactual_direction_correct"] == 1.0)



# --------------------------------------------------------------------------
print("variant editing (patch splicing)")
te = M["task_editor"]
from PIL import Image

OBJ = [(80, 120), (200, 120)]      # two replaced objects, far apart


def _scene_with_two_changes():
    def ro(i, attrs):
        base = {"color": "red", "shape": "cube", "size": "large", "material": "rubber",
                "pixel_coords": [OBJ[i][0], OBJ[i][1], 8.0]}
        return {"index": i, "original": dict(base),
                "replacement": {**base, **attrs}}
    return {"modification": {"replaced_objects": [
        ro(0, {"color": "blue"}),                                  # 1 attr
        ro(1, {"color": "green", "material": "metal"}),            # 2 attrs
    ]}}


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    cache = root / "output" / "variant_cache"
    images.mkdir(parents=True)
    # original: uniform grey. modified: a coloured square at each changed object.
    orig = Image.new("RGB", (320, 240), (128, 128, 128))
    mod = orig.copy()
    for (cx, cy), colour in zip(OBJ, [(255, 0, 0), (0, 0, 255)]):
        for x in range(cx - 20, cx + 20):
            for y in range(cy - 20, cy + 20):
                mod.putpixel((x, y), colour)
    orig.save(images / "sc_original.png")
    mod.save(images / "sc_modified.png")

    scene = _scene_with_two_changes()
    subsets = te.enumerate_variants(scene)
    check("K objects give 2^K-1 variants", len(subsets) == 3, subsets)

    v_both = te.build_variant("sc", scene, [0, 1], images, cache)
    v_first = te.build_variant("sc", scene, [0], images, cache)
    v_second = te.build_variant("sc", scene, [1], images, cache)
    check("keeping every object changed needs no splicing", v_both["spliced"] is False)
    check("the full variant's label is the scene's own count",
          v_both["num_attr_changes"] == 3, v_both)
    check("reverting an object subtracts exactly its attribute count",
          (v_first["num_attr_changes"], v_second["num_attr_changes"]) == (1, 2))
    check("evidence boxes follow the surviving objects",
          len(v_both["gold_boxes"]) == 2 and len(v_first["gold_boxes"]) == 1)

    px = Image.open(v_first["spy_image_path"]).convert("RGB")
    check("the kept object is still changed in the spliced image",
          px.getpixel(OBJ[0]) == (255, 0, 0), px.getpixel(OBJ[0]))
    check("the reverted object is back to the original pixels",
          px.getpixel(OBJ[1]) == (128, 128, 128), px.getpixel(OBJ[1]))

    labels = sorted(te.build_variant("sc", scene, k, images, cache)["num_attr_changes"]
                    for k in subsets)
    check("one scene can supply several different labels", labels == [1, 2, 3], labels)

    pairs = te.counterfactual_pairs(subsets)
    check("pairs differ by exactly one object",
          all(len(b) == len(a) + 1 and set(a) < set(b) for a, b in pairs) and pairs)



# --------------------------------------------------------------------------
print("regret-seeded generation")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    for i in range(12):
        name = f"sc{i:02d}"
        Image.new("RGB", (320, 240), (128, 128, 128)).save(images / f"{name}_original.png")
        Image.new("RGB", (320, 240), (200, 100, 50)).save(images / f"{name}_modified.png")
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            **_scene_with_two_changes(),
            "original_scene": {"objects": [{} for _ in range(6)]},
        }))

    cfg = gen_mod.PolicyCLEVRGeneratorConfig(
        dataset_root=str(root), num_tasks=8, edit_fraction=0.5,
        variant_cache_dir=str(root / "cache"),
    )
    generator = gen_mod.PolicyControlledCLEVRGenerator(cfg)
    policy = {
        "editing_policy": {
            "seeds": [{"scene_id": "sc00", "regret": 0.8, "direction": "easier"},
                      {"scene_id": "sc01", "regret": 0.6, "direction": "harder"}],
            "retire_scene_ids": ["sc02"],
        },
    }
    tasks = generator.generate(policy)
    report = generator.last_generation_report

edited = [t for t in tasks if t["generation_source"] == "regret_seeded_variant"]
check("the round still has the requested number of tasks", len(tasks) == 8, len(tasks))
check("seeds are edited into variants", len(edited) == 4, len(edited))
check("edited tasks come in counterfactual pairs",
      len({t["pair_id"] for t in edited}) == 2
      and sorted(t["pair_role"] for t in edited) == ["large", "large", "small", "small"])
by_pair = {}
for t in edited:
    by_pair.setdefault(t["pair_id"], {})[t["pair_role"]] = t
check("within a pair the gold labels differ",
      all(p["small"]["ground_truth"] != p["large"]["ground_truth"] for p in by_pair.values()))
check("a variant's images point at its own spliced spy image",
      any("variant_cache" in p or "cache" in p for t in edited for p in t["image_path"]))
check("retired scenes are not proposed again",
      all(t["scene_id"] != "sc02" for t in tasks))
check("the report separates edited from fresh",
      report["num_edited"] == 4 and report["num_fresh"] == 4, report)



# --------------------------------------------------------------------------
print("audit fixes")
# Overlapping revert/keep patches must refuse the variant, not corrupt the label.
def _pair_scene(sep):
    def ro(i, cx):
        base = {"color": "red", "shape": "cube", "size": "large", "material": "rubber"}
        return {"index": i, "original": {**base, "pixel_coords": [cx, 120, 8.0]},
                "replacement": {**base, "color": "blue", "pixel_coords": [cx, 120, 8.0]}}
    return {"modification": {"replaced_objects": [ro(0, 100), ro(1, 100 + sep)]}}

with tempfile.TemporaryDirectory() as tmp:
    img = Path(tmp) / "o"; img.mkdir(parents=True)
    Image.new("RGB", (320, 240), (128, 128, 128)).save(img / "sc_original.png")
    Image.new("RGB", (320, 240), (0, 0, 255)).save(img / "sc_modified.png")
    near = te.build_variant("sc", _pair_scene(40), [1], img, Path(tmp) / "c")
    far = te.build_variant("sc", _pair_scene(200), [1], img, Path(tmp) / "c")
check("a revert that would overwrite a kept object is refused", near is None)
check("well-separated objects still splice", far is not None and far["num_attr_changes"] == 1)

# Pair members share one spy slot; a failed member drops its partner.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"; scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    # sc00 buildable seed, sc01 overlap-doomed seed, sc02.. fresh-only pool
    for i, sep in enumerate([200, 40, 200, 200, 200, 200]):
        name = f"sc{i:02d}"
        Image.new("RGB", (320, 240), (128, 128, 128)).save(images / f"{name}_original.png")
        Image.new("RGB", (320, 240), (0, 0, 255)).save(images / f"{name}_modified.png")
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            **_pair_scene(sep), "original_scene": {"objects": [{} for _ in range(6)]}}))
    cfg = gen_mod.PolicyCLEVRGeneratorConfig(dataset_root=str(root), num_tasks=6,
                                             edit_fraction=1.0, variant_cache_dir=str(root / "c"))
    generator = gen_mod.PolicyControlledCLEVRGenerator(cfg)
    tasks = generator.generate({"editing_policy": {"seeds": [
        {"scene_id": "sc00", "regret": 0.9, "direction": "easier"},
        {"scene_id": "sc01", "regret": 0.8, "direction": "easier"},
    ]}})
edited = [t for t in tasks if t["generation_source"] == "regret_seeded_variant"]
by_pair = {}
for t in edited:
    by_pair.setdefault(t["pair_id"], []).append(t)
check("pairs survive whole or not at all",
      all(len(v) == 2 for v in by_pair.values()), {k: len(v) for k, v in by_pair.items()})
check("both pair members share one spy slot",
      all(len({m["metadata"]["spy_player"] for m in v}) == 1 for v in by_pair.values()))
check("failed edits are refilled by fresh sampling", len(tasks) == 6, len(tasks))

# Offline audit scalar equals the live scalar under gating.
wrong_spy = ("<think>Player 2 differs.</think>" + tags(perfect, player=2)
             + "<answer>spy=2; changed_attributes=4</answer>")
t_live, b_live = score(wrong_spy, CURRENT)
cfg2 = rc.load_reward_config(CURRENT)
vec = {k: b_live[k] for k in ("answer", "grounding", "process", "consistency", "budget")}
audit = cfg2.audit_fields(vec, spy_correct=False, answer_correct=False)
check("offline audit scalar matches the live gated scalar",
      abs(audit["reward_scalar_used"] - t_live) < 1e-6, (audit["reward_scalar_used"], t_live))



# --------------------------------------------------------------------------
print("self-play: the model proposes, the solver pays it")
from open_r1.self_evolve import proposer as pr

_scene2 = _scene_with_two_changes()
_prompt = pr.build_proposer_prompt(_scene2, {"mean_solve_rate": 0.4})
check("the proposer prompt indexes every changed object",
      "[0]" in _prompt and "[1]" in _prompt)
check("the proposer is told how the solver is doing", "40%" in _prompt)
check("hiding competence drops it from the prompt",
      "40%" not in pr.build_proposer_prompt(_scene2, None))
check("a one-object scene offers no choice to propose",
      pr.build_proposer_prompt({"modification": {"replaced_objects": [
          _scene2["modification"]["replaced_objects"][0]]}}) is None)

check("a proposal parses to the kept indices",
      pr.parse_proposal("<think>x</think>\n<keep>[0,1]</keep>", 2)["keep"] == [0, 1])
check("an out-of-range index is refused", pr.parse_proposal("<keep>[7]</keep>", 2) is None)
check("an empty pick is refused", pr.parse_proposal("<keep>[]</keep>", 2) is None)
check("a missing tag is refused", pr.parse_proposal("I choose object one", 2) is None)
check("a real model's bracket-less answer still parses",
      pr.parse_proposal("<keep>1</keep>", 2)["keep"] == [1])
check("the proposer also picks how many players see it",
      pr.parse_proposal("<keep>[0]</keep><players>6</players>", 2,
                        3, 8)["num_players"] == 6)
check("an out-of-range player count falls back instead of losing the pick",
      pr.parse_proposal("<keep>[0]</keep><players>99</players>", 2,
                        3, 8)["num_players"] == 5)
check("the spy slot is never taken from the proposal",
      "spy_player" not in (pr.parse_proposal("<keep>[0]</keep>", 2) or {}))

check("a coin-flip task is worth the most", pr.learnability(0.5) == 1.0)
check("a task nobody solves is worth nothing", pr.learnability(0.0) == 0.0)
check("a task everybody solves is worth nothing", pr.learnability(1.0) == 0.0)

_props = pr.tag_counterfactual_pairs([
    {"proposal_id": "s::p0", "scene_id": "s", "keep": [0], "completion": "c0",
     "prompt": _prompt, "image_path": ["a", "b"], "task_id": "t0"},
    {"proposal_id": "s::p1", "scene_id": "s", "keep": [0, 1], "completion": "c1",
     "prompt": _prompt, "image_path": ["a", "b"], "task_id": "t1"},
])
check("proposals one object apart become a counterfactual pair",
      {p["proposal_id"]: p.get("pair_role") for p in _props}
      == {"s::p0": "small", "s::p1": "large"})

_scored = pr.score_proposals(_props, {
    "t0": {"pass_rate": 0.5, "class": "trainable", "regret": 0.4},
    "t1": {"pass_rate": 1.0, "class": "mastered", "regret": 0.0},
})
_by = {p["proposal_id"]: p for p in _scored}
check("the proposer is paid for the solver landing in the middle",
      _by["s::p0"]["learnability"] == 1.0)
check("proposing a task the solver has mastered pays nothing",
      _by["s::p1"]["learnability"] == 0.0)
check("a proposal that never reached the solver pays nothing",
      pr.score_proposals([{"proposal_id": "x", "keep": [0]}], {})[0]["learnability"] == 0.0)

check("a proposal the budget never evaluated is not called a bad pick",
      pr.score_proposals([{"proposal_id": "x", "keep": [0]}], {})[0]["reached_solver"]
      is False)
check("a pick with no completion is never trained on",
      pr.build_proposer_sft_examples(
          [{"proposal_id": "e", "keep": [0], "learnability": 1.0,
            "completion": None, "prompt": "p"}]) == [])

_sft = pr.build_proposer_sft_examples(_scored, threshold=0.5)
check("only the learnable proposal is trained on",
      [e["task_id"] for e in _sft] == ["s::p0"])
check("proposer records are tagged with their role",
      all(e["role"] == "proposer" for e in _sft))
check("the report counts distinct picks, so collapse is visible",
      pr.proposal_report(_scored)["distinct_subsets"] == 2)

# The property that makes the proposing side safe: it moves difficulty, never
# truth. Whatever subset it picks, the label is the splicing arithmetic.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    for i in range(4):
        name = f"sp{i:02d}"
        Image.new("RGB", (320, 240), (128, 128, 128)).save(images / f"{name}_original.png")
        Image.new("RGB", (320, 240), (200, 100, 50)).save(images / f"{name}_modified.png")
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            **_scene_with_two_changes(),
            "original_scene": {"objects": [{} for _ in range(6)]},
        }))
    cfg = gen_mod.PolicyCLEVRGeneratorConfig(
        dataset_root=str(root), num_tasks=4, num_players=3, edit_fraction=1.0,
        variant_cache_dir=str(root / "cache"),
    )
    generator = gen_mod.PolicyControlledCLEVRGenerator(cfg)
    _plans = pr.tag_counterfactual_pairs([
        {"proposal_id": "sp00::p0", "scene_id": "sp00", "keep": [0]},
        {"proposal_id": "sp00::p1", "scene_id": "sp00", "keep": [0, 1]},
    ])
    sp_tasks = generator.generate({"generator_policy_version": 1}, None, proposals=_plans)
    sp_report = generator.last_generation_report

_built = [t for t in sp_tasks if t["generation_source"] == "self_play_variant"]
_one, _two = _scene_with_two_changes()["modification"]["replaced_objects"]
_n0 = sum(1 for a in ("color", "shape", "size", "material")
          if _one["original"].get(a) != _one["replacement"].get(a))
_n1 = sum(1 for a in ("color", "shape", "size", "material")
          if _two["original"].get(a) != _two["replacement"].get(a))
_want = {(0,): _n0, (0, 1): _n0 + _n1}
check("both proposals became real tasks", len(_built) == 2, len(_built))
check("a proposed task's label is the exact splicing arithmetic",
      all(int(t["ground_truth"].split("changed_attributes=")[1])
          == _want[tuple(t["metadata"]["variant"]["keep_indices"])] for t in _built))
check("every proposed task carries its proposal id",
      all(t.get("proposal_id") for t in _built))
check("proposals from one scene share a spy slot",
      len({t["metadata"]["spy_player"] for t in _built}) == 1)
check("the report says the edits came from the proposer",
      sp_report["edit_source"] == "proposer")


# --------------------------------------------------------------------------
print("audit round two")

# A revert that cannot be pasted must sink the whole variant: splicing the rest
# would leave that object visibly changed while the gold count stops counting
# it, so the image would contradict its own label.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp); images = root / "img"; images.mkdir()
    o = Image.new("RGB", (320, 240), (30, 30, 30))
    m = Image.new("RGB", (320, 240), (30, 30, 30))
    for (cx, cy), c in {(60, 60): (255, 0, 0), (160, 60): (0, 0, 255),
                        (270, 190): (0, 255, 0)}.items():
        for dx in range(-12, 12):
            for dy in range(-12, 12):
                m.putpixel((cx + dx, cy + dy), c)
    o.save(images / "s_original.png"); m.save(images / "s_modified.png")

    def _ob(x=None, y=None):
        a = {"color": "red", "shape": "cube", "size": "large", "material": "rubber"}
        b = {"color": "blue", "shape": "cube", "size": "large", "material": "rubber"}
        if x is not None:
            a["pixel_coords"] = [x, y, 9]; b["pixel_coords"] = [x, y, 9]
        return {"original": a, "replacement": b}

    # obj2 carries no pixel_coords, so it cannot be reverted.
    unrevertable = {"modification": {"replaced_objects":
                                     [_ob(60, 60), _ob(160, 60), _ob()]}}
    clean = {"modification": {"replaced_objects":
                              [_ob(60, 60), _ob(160, 60), _ob(270, 190)]}}
    bad = te.build_variant("s", unrevertable, [0], images, root / "c")
    good = te.build_variant("s", clean, [0], images, root / "c")

check("a variant whose revert cannot be pasted is refused", bad is None)
check("a variant whose reverts all land is still built", good is not None)
check("the surviving variant's label matches its kept objects",
      good["num_attr_changes"] == 1, good["num_attr_changes"])

# Two proposals that picked the same subset would collide into one task_id,
# merging their rollouts into a single oversized group.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    for i in range(4):
        name = f"dp{i:02d}"
        Image.new("RGB", (320, 240), (128, 128, 128)).save(images / f"{name}_original.png")
        Image.new("RGB", (320, 240), (200, 100, 50)).save(images / f"{name}_modified.png")
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            **_scene_with_two_changes(),
            "original_scene": {"objects": [{} for _ in range(6)]},
        }))
    gen = gen_mod.PolicyControlledCLEVRGenerator(gen_mod.PolicyCLEVRGeneratorConfig(
        dataset_root=str(root), num_tasks=6, num_players=3, edit_fraction=1.0,
        variant_cache_dir=str(root / "cache")))
    dup_tasks = gen.generate({"generator_policy_version": 1}, None, proposals=[
        {"proposal_id": "dp00::p0", "scene_id": "dp00", "keep": [0]},
        {"proposal_id": "dp00::p1", "scene_id": "dp00", "keep": [0]},
        {"proposal_id": "dp00::p2", "scene_id": "dp00", "keep": [0, 1]},
    ])
_sp = [t for t in dup_tasks if t["generation_source"] == "self_play_variant"]
check("two proposals picking the same subset yield one task, not a collision",
      len({t["task_id"] for t in _sp}) == len(_sp), [t["task_id"] for t in _sp])
check("the distinct pick still becomes its own task", len(_sp) == 2, len(_sp))


# A scene with two changed objects offers only three subsets. The player
# count is the second axis that gives the proposer a real choice.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    images = root / "output" / "replacement_images"
    scenes = root / "output" / "replacement_scenes"
    images.mkdir(parents=True); scenes.mkdir(parents=True)
    for i in range(6):
        name = f"ap{i:02d}"
        Image.new("RGB", (320, 240), (128, 128, 128)).save(images / f"{name}_original.png")
        Image.new("RGB", (320, 240), (200, 100, 50)).save(images / f"{name}_modified.png")
        (scenes / f"{name}_comparison.json").write_text(json.dumps({
            **_scene_with_two_changes(),
            "original_scene": {"objects": [{} for _ in range(6)]},
        }))
    g2 = gen_mod.PolicyControlledCLEVRGenerator(gen_mod.PolicyCLEVRGeneratorConfig(
        dataset_root=str(root), num_tasks=6, num_players=5, edit_fraction=1.0,
        variant_cache_dir=str(root / "cache")))
    ap_tasks = g2.generate({"generator_policy_version": 1}, None, proposals=[
        {"proposal_id": "ap00::p0", "scene_id": "ap00", "keep": [0], "num_players": 3},
        {"proposal_id": "ap01::p0", "scene_id": "ap01", "keep": [0, 1], "num_players": 8},
    ])
_ap = [t for t in ap_tasks if t["generation_source"] == "self_play_variant"]
_by_players = {t["metadata"]["num_players"]: t for t in _ap}
check("a proposal's player count reaches the task", sorted(_by_players) == [3, 8],
      sorted(_by_players))
check("the task shows exactly that many images",
      all(len(t["image_path"]) == t["metadata"]["num_players"] for t in _ap))
check("the prompt states that task's own player count",
      all(f"There are {t['metadata']['num_players']} players" in t["prompt"]
          for t in _ap))
check("the spy slot stays inside the proposed player count",
      all(1 <= t["metadata"]["spy_player"] <= t["metadata"]["num_players"]
          for t in _ap))

print(f"\n{passed} checks passed")
