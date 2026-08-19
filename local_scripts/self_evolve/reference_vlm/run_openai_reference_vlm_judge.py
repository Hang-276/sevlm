#!/usr/bin/env python
"""
GPT-4o Reference VLM judge over candidate tasks (full-loop front stage).

This is the **Reference VLM** stage of the Vision-Zero self-evolving loop:

    Generator -> candidate tasks -> [GPT-4o Reference VLM] -> accepted tasks -> Solver VLM

The Reference VLM evaluates each candidate task *before* it ever reaches the
Solver. It decides whether the task is solvable, appropriately difficult,
unambiguous, and verifiable from visual evidence, and it emits a reasoning
budget plus reference reasoning steps that the downstream reward shaping reads.

Provider support (the Reference VLM is a REQUIRED module, not optional):
  - ``--reference-provider openrouter`` (default): OpenRouter relay, base URL
    ``https://openrouter.ai/api/v1``, model slug ``openai/gpt-4o``, key from
    ``OPENROUTER_API_KEY``.
  - ``--reference-provider openai``: official OpenAI API, base URL
    ``https://api.openai.com/v1``, key from ``OPENAI_API_KEY``.

The API key is read from the
environment only — never from code, never from a checked-in file, and is NEVER
printed (no Authorization header is logged; errors print type/message only).

USAGE (no API, structure check)::

    python run_openai_reference_vlm_judge.py --help
    python run_openai_reference_vlm_judge.py --task-jsonl tasks.jsonl \
        --dataset-root /path/to/clevr --output-dir out/ref --dry-run

USAGE (real OpenRouter GPT-4o reference smoke, key required)::

    read -s OPENROUTER_API_KEY; export OPENROUTER_API_KEY
    python run_openai_reference_vlm_judge.py --task-jsonl tasks.jsonl \
        --dataset-root /path/to/clevr --output-dir out/ref \
        --reference-provider openrouter --reference-model openai/gpt-4o \
        --reference-base-url https://openrouter.ai/api/v1 --max-tasks 3

Outputs (under --output-dir):
  openai_reference_judgments.jsonl   — one judgment per candidate task
  openai_reference_summary.json      — aggregate accept/reject stats

This script does NOT train and does NOT modify the Vision-Zero trainer.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"


def _resolve_provider_config(
    provider: str,
    base_url: Optional[str],
    model: Optional[str],
) -> Dict[str, Optional[str]]:
    """Resolve (api_key_env, base_url, model, api_key) for the provider.

    Never returns or logs the key value beyond what the caller needs; the key
    itself is only read here and passed straight to the client.
    """
    if provider == "openrouter":
        key_env = "OPENROUTER_API_KEY"
        resolved_base = base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL_DEFAULT
        resolved_model = model or os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o"
    elif provider == "openai":
        key_env = "OPENAI_API_KEY"
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL_DEFAULT
        resolved_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"
    else:
        raise SystemExit(f"Unknown --reference-provider: {provider}")
    return {
        "key_env": key_env,
        "base_url": resolved_base,
        "model": resolved_model,
        "api_key": os.environ.get(key_env),
    }


JUDGMENT_FIELDS = [
    "task_id",
    "model",
    "reference_provider",
    "input_mode",
    "is_solvable",
    "accepted",
    "reject_reason",
    "difficulty_level",
    "difficulty_score",
    "solvability_score",
    "visual_grounding_score",
    "ambiguity_score",
    "ambiguity_flag",
    "visual_verifiability",
    "reasoning_budget_steps",
    "reasoning_budget_tokens",
    "required_visual_evidence",
    "reference_reasoning_steps",
    "reference_answer_check",
    "reference_feedback",
    "generator_feedback",
    "suggested_failure_focus",
]

SYSTEM_PROMPT = (
    "You are a Reference VLM judge for a visual-reasoning training loop. You are "
    "shown the candidate task's player images and the task prompt. You do NOT "
    "solve the task to produce a training answer — you JUDGE whether the task is "
    "suitable for training the Solver, and you provide reference guidance.\n\n"
    "Decide:\n"
    "  1. Is the task solvable from the visual evidence?\n"
    "  2. Is its difficulty appropriate (not impossibly hard)?\n"
    "  3. Is it free of ambiguity (a single defensible answer)?\n"
    "  4. Can it be verified from visual evidence (not guesswork)?\n"
    "  5. What is a reasonable reasoning budget (steps and tokens)?\n"
    "  6. What visual evidence is required, and what are good reference reasoning "
    "steps?\n\n"
    "Reject (accepted=false) when the task is over-hard, ambiguous, or not "
    "visually verifiable, and give a reject_reason. Otherwise accepted=true.\n\n"
    "Return ONLY a JSON object with keys: is_solvable (bool), accepted (bool), "
    "reject_reason (string, empty when accepted), difficulty_level "
    "('easy'|'medium'|'hard'), difficulty_score (number 1-5), solvability_score "
    "(number 0-1), visual_grounding_score (number 0-1), ambiguity_score "
    "(number 0-1), ambiguity_flag (bool), visual_verifiability (number 0-1), "
    "reasoning_budget_steps (int), reasoning_budget_tokens (int), "
    "required_visual_evidence (list of strings), reference_reasoning_steps "
    "(list of strings), reference_answer_check (string), reference_feedback "
    "(string), generator_feedback (object with keys avoid_scene_id, "
    "suggested_difficulty), suggested_failure_focus (list of strings). No prose "
    "outside the JSON."
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_image_paths(task: Dict[str, Any], dataset_root: Optional[Path]) -> List[str]:
    images = task.get("image_path") or task.get("images") or []
    if isinstance(images, str):
        images = [images]
    resolved: List[str] = []
    for ip in images:
        p = Path(ip)
        if not p.is_absolute() and dataset_root is not None:
            cand = dataset_root / ip
            if cand.is_file():
                p = cand
        resolved.append(str(p))
    return resolved


def _encode_image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _build_messages(task: Dict[str, Any], image_paths: List[str]) -> List[Dict[str, Any]]:
    problem = str(task.get("prompt") or task.get("problem") or "")
    answer = str(task.get("answer") or task.get("solution") or "")
    content: List[Dict[str, Any]] = []
    for ip in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_data_url(ip)}})
    content.append({
        "type": "text",
        "text": (
            f"Candidate task:\n{problem}\n\n"
            f"Gold answer (for your judgment only, do not reveal): {answer}\n\n"
            "Return the JSON judgment."
        ),
    })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _parse_judgment(
    data: Dict[str, Any], task_id: str, model: str, provider: str = "openrouter"
) -> Dict[str, Any]:
    accepted = bool(data.get("accepted", data.get("is_solvable", True)))
    reject_reason = str(data.get("reject_reason", "") or ("" if accepted else "rejected"))
    difficulty_level = str(data.get("difficulty_level", "medium"))
    # Ensure generator_feedback carries suggested_difficulty + reason so the
    # policy update can consume them. When the model
    # rejects a task as too hard, suggest an EASIER difficulty.
    gen_fb = data.get("generator_feedback") if isinstance(data.get("generator_feedback"), dict) else {}
    too_hard = (not accepted) and ("too" in reject_reason.lower() and (
        "hard" in reject_reason.lower() or "difficult" in reject_reason.lower()))
    if "suggested_difficulty" not in gen_fb:
        explicit = data.get("suggested_difficulty")
        if explicit:
            gen_fb["suggested_difficulty"] = str(explicit)
        elif too_hard:
            gen_fb["suggested_difficulty"] = "easy" if difficulty_level == "easy" else "medium"
        else:
            gen_fb["suggested_difficulty"] = difficulty_level
    if "reason" not in gen_fb:
        gen_fb["reason"] = "reduce_difficulty" if too_hard else ""
    return {
        "task_id": task_id,
        "model": model,
        "reference_provider": provider,
        "input_mode": "vision",
        "is_solvable": bool(data.get("is_solvable", True)),
        "accepted": accepted,
        "reject_reason": reject_reason,
        "difficulty_level": difficulty_level,
        "difficulty_score": float(data.get("difficulty_score", 3)),
        "solvability_score": float(data.get("solvability_score", 1.0 if accepted else 0.4)),
        "visual_grounding_score": float(data.get("visual_grounding_score", data.get("visual_verifiability", 0.7))),
        "ambiguity_score": float(data.get("ambiguity_score", 0.7 if data.get("ambiguity_flag") else 0.2)),
        "ambiguity_flag": bool(data.get("ambiguity_flag", False)),
        "visual_verifiability": float(data.get("visual_verifiability", 1.0)),
        "reasoning_budget_steps": int(data.get("reasoning_budget_steps", 4)),
        "reasoning_budget_tokens": int(data.get("reasoning_budget_tokens", 160)),
        "required_visual_evidence": list(data.get("required_visual_evidence", [])),
        "reference_reasoning_steps": list(data.get("reference_reasoning_steps", [])),
        "reference_answer_check": str(data.get("reference_answer_check", "")),
        "reference_feedback": str(data.get("reference_feedback", "")),
        "suggested_difficulty": gen_fb.get("suggested_difficulty"),
        "generator_feedback": gen_fb,
        "suggested_failure_focus": list(data.get("suggested_failure_focus", [])),
        "reference_source": f"{provider}_reference_vlm",
    }


def _call_gpt4o(
    task: Dict[str, Any],
    image_paths: List[str],
    model: str,
    api_key: str,
    base_url: str,
    max_retries: int,
    provider: str = "openrouter",
) -> Dict[str, Any]:
    from openai import OpenAI  # imported lazily — never at module import time

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(task, image_paths)
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            return _parse_judgment(data, str(task.get("task_id", "")), model, provider)
        except Exception as exc:  # noqa: BLE001
            # Print error TYPE and short message only — never the key/auth header.
            last_err = f"{type(exc).__name__}: {str(exc)[:160]}"
            if attempt >= max_retries:
                break
    raise RuntimeError(
        f"Reference VLM API failure (provider={provider}, model={model}, "
        f"task={task.get('task_id')}): {last_err}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-4o Reference VLM judge over candidate tasks (full-loop front stage)."
    )
    parser.add_argument("--task-jsonl", required=True,
                        help="Candidate task manifest JSONL (from the generator / manifest builder).")
    parser.add_argument("--dataset-root", default=None,
                        help="Image/dataset root used to resolve relative image_path entries.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-provider", default="openrouter",
                        choices=["openrouter", "openai"],
                        help="Reference VLM provider (default: openrouter).")
    parser.add_argument("--reference-model", default=None,
                        help="Reference VLM model slug (default: openai/gpt-4o for "
                             "openrouter, gpt-4o for openai).")
    parser.add_argument("--reference-base-url", default=None,
                        help="Reference VLM base URL (default: provider default).")
    parser.add_argument("--model", dest="model_alias", default=None,
                        help="DEPRECATED alias for --reference-model.")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of candidate tasks judged (API budget cap).")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not call the API; print what would be judged.")
    args = parser.parse_args()

    # Load .env (gitignored) if present — shell env always wins (override=False).
    try:
        from open_r1.self_evolve.dotenv_loader import load_dotenv
        loaded = load_dotenv()
        if loaded:
            print(f"[dotenv] loaded keys from .env: {sorted(loaded)}")
    except Exception:
        pass

    task_path = Path(args.task_jsonl)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    output_dir = Path(args.output_dir)

    # Resolve provider config (key env / base url / model). Backward-compat:
    # --model maps to --reference-model when the latter is unset.
    model_arg = args.reference_model or args.model_alias
    if args.model_alias and not args.reference_model:
        print(f"[compat] --model is deprecated; using --reference-model={args.model_alias}")
    cfg = _resolve_provider_config(args.reference_provider, args.reference_base_url, model_arg)
    provider = args.reference_provider
    base_url = cfg["base_url"]
    model = cfg["model"]

    print(f"{'='*60}\n  REFERENCE VLM JUDGE (front stage)\n{'='*60}")
    print(f"  Mode:        {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"  Provider:    {provider}")
    print(f"  Task JSONL:  {task_path}")
    print(f"  Dataset:     {dataset_root}")
    print(f"  Model:       {model}")
    print(f"  Base URL:    {base_url}")
    print(f"  Key env:     {cfg['key_env']} ({'present' if cfg['api_key'] else 'MISSING'})")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}")

    if args.dry_run:
        if task_path.is_file():
            tasks = _read_jsonl(task_path)
            if args.max_tasks is not None:
                tasks = tasks[: args.max_tasks]
            print(f"[DRY-RUN] Would judge {len(tasks)} candidate task(s) with {model}.")
            for t in tasks[:3]:
                imgs = _resolve_image_paths(t, dataset_root)
                print(f"  - task_id={t.get('task_id')} images={len(imgs)}")
        else:
            print(f"[DRY-RUN] Task JSONL not found (would be produced by the generator): {task_path}")
        print("[DRY-RUN] No API call made. No output written.")
        sys.exit(0)

    # --- LIVE path: key is mandatory, fail fast ---
    api_key = cfg["api_key"]
    if not api_key:
        raise SystemExit(
            f"{cfg['key_env']} is required for the live Reference VLM "
            f"(provider={provider})."
        )

    if not task_path.is_file():
        raise SystemExit(f"Task JSONL not found: {task_path}")

    tasks = _read_jsonl(task_path)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "openai_reference_judgments.jsonl"
    summary_path = output_dir / "openai_reference_summary.json"

    accepted = 0
    rejected = 0
    reject_reasons: Dict[str, int] = {}
    with judgments_path.open("w", encoding="utf-8") as out:
        for task in tasks:
            image_paths = _resolve_image_paths(task, dataset_root)
            judgment = _call_gpt4o(
                task, image_paths, model, api_key, base_url, args.max_retries, provider
            )
            out.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            if judgment["accepted"]:
                accepted += 1
            else:
                rejected += 1
                rr = judgment.get("reject_reason") or "unspecified"
                reject_reasons[rr] = reject_reasons.get(rr, 0) + 1

    summary = {
        "reference_provider": provider,
        "model": model,
        "input_mode": "vision",
        "base_url": base_url,
        "num_tasks": len(tasks),
        "num_accepted": accepted,
        "num_rejected": rejected,
        "reject_reason_distribution": reject_reasons,
        "judgments_path": str(judgments_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nJudged {len(tasks)} task(s): accepted={accepted} rejected={rejected}")
    print(f"  Judgments → {judgments_path}")
    print(f"  Summary   → {summary_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
