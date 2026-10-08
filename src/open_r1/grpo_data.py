"""Shared prompt and reward-context conversion for GRPO records."""

import json
from pathlib import Path

from open_r1.self_evolve.verified_qa_reward import render_training_question


def normalize_reward_context(context):
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid self_evolve reward context") from exc
    if context is not None and not isinstance(context, dict):
        raise ValueError("self_evolve reward context must be an object")
    return context


def prepare_grpo_sample(example, question_prompt, *, check_images=False):
    context = normalize_reward_context(example.get("self_evolve"))
    problem = str(example.get("problem", ""))
    solution = str(example.get("solution", ""))
    if not solution.startswith("<answer>"):
        solution = f"<answer> {solution} </answer>"
    processed = {
        "problem": problem,
        "solution": solution,
        "accu_reward_method": example.get("accu_reward_method", "default"),
    }
    if "self_evolve" in example:
        processed["self_evolve"] = json.dumps(context, ensure_ascii=False)
    content = []
    images = example.get("image_path")
    if images is not None:
        if not isinstance(images, list) or any(not isinstance(p, str) or not p for p in images):
            raise ValueError("image_path must be a list of nonempty paths")
        if check_images and any(not Path(p).is_file() for p in images):
            raise ValueError(f"Image paths do not exist: {images}")
        processed["image_path"] = list(images)
        content.extend({"type": "image", "text": None} for _ in images)
    content.append({"type": "text", "text": render_training_question(problem, context, question_prompt)})
    processed["prompt"] = [{"role": "user", "content": content}]
    return processed
