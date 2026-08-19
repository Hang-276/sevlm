"""Evidence-box parsing and IoU grounding.

Parses the boxes a solver emits, matches them against the gold evidence boxes
(recall or box-F1), and turns that into the grounding reward.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# Two syntaxes are accepted: <bbox player="1">[x1,y1,x2,y2]</bbox> and
# Qwen2.5-VL's native {"bbox_2d": [...]}. ``coordinate_mode`` picks normalized
# vs absolute pixels.

_BBOX_RE = re.compile(
    r"<bbox(?P<attrs>[^>]*)>\s*\[?\s*"
    r"(?P<x1>-?\d*\.?\d+)\s*,\s*(?P<y1>-?\d*\.?\d+)\s*,\s*"
    r"(?P<x2>-?\d*\.?\d+)\s*,\s*(?P<y2>-?\d*\.?\d+)\s*\]?\s*</bbox>",
    re.IGNORECASE | re.DOTALL,
)
_PLAYER_ATTR_RE = re.compile(r"player\s*=\s*[\"']?([^\"'>\s]+)[\"']?", re.IGNORECASE)

# Qwen2.5-VL's native syntax, in absolute pixels — a format the solver was
# pretrained on rather than one it has to learn from a single example.
_BBOX_2D_RE = re.compile(
    r"\{(?P<obj>[^{}]*?\"bbox_2d\"\s*:\s*\[\s*"
    r"(?P<x1>-?\d*\.?\d+)\s*,\s*(?P<y1>-?\d*\.?\d+)\s*,\s*"
    r"(?P<x2>-?\d*\.?\d+)\s*,\s*(?P<y2>-?\d*\.?\d+)\s*\][^{}]*?)\}",
    re.IGNORECASE | re.DOTALL,
)
_PLAYER_JSON_RE = re.compile(r"\"player\"\s*:\s*\"?(\d+)\"?", re.IGNORECASE)


def _resolve_coordinate_scale(
    coords: List[float],
    coordinate_mode: str,
    image_width: Optional[int],
    image_height: Optional[int],
    is_json: bool,
) -> Optional[List[float]]:
    """Return coords normalized to [0,1], or None if out of range for the mode."""
    mode = coordinate_mode
    if mode == "auto":
        mode = "pixel" if (is_json or max(coords) > 1.0) else "normalized"
    if mode == "pixel":
        if not (image_width and image_height):
            return None
        x1, y1, x2, y2 = coords
        out = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
    else:
        out = list(coords)
    return out if all(0.0 <= v <= 1.0 for v in out) else None


def parse_evidence_boxes(
    completion: Optional[str],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    valid_player_ids: Optional[List[int]] = None,
    coordinate_mode: str = "normalized",
) -> Dict[str, Any]:
    """Parse evidence boxes from a completion (``<bbox>`` tags or ``bbox_2d`` JSON).

    ``coordinate_mode`` selects how the numbers are read: ``normalized`` ([0,1]),
    ``pixel`` (absolute, Qwen2.5-VL's native convention) or ``auto``.
    ``valid_player_ids`` restricts the accepted player id; a bad id loses the
    player-id credit but keeps the box's IoU signal.

    Returns boxes_normalized / boxes_pixel / players / first_player /
    num_found / num_valid / bbox_valid / bbox_parse_error / errors.
    """
    out: Dict[str, Any] = {
        "boxes_normalized": [],
        "boxes_pixel": [],
        "players": [],
        "first_player": None,
        "num_found": 0,
        "num_valid": 0,
        "bbox_valid": False,
        "bbox_parse_error": None,
        "errors": [],
    }
    if not completion:
        out["bbox_parse_error"] = "no_completion"
        return out

    # (match, raw_player_string|None, is_json)
    matches: List[Tuple[Any, Optional[str], bool]] = [
        (m, (_PLAYER_ATTR_RE.search(m.group("attrs") or "") or [None, None])[1]
            if _PLAYER_ATTR_RE.search(m.group("attrs") or "") else None, False)
        for m in _BBOX_RE.finditer(completion)
    ]
    for m in _BBOX_2D_RE.finditer(completion):
        pm = _PLAYER_JSON_RE.search(m.group("obj") or "")
        matches.append((m, pm.group(1) if pm else None, True))

    out["num_found"] = len(matches)
    if not matches:
        out["bbox_parse_error"] = "no_bbox_tag"
        return out

    valid_set = set(int(p) for p in valid_player_ids) if valid_player_ids else None

    errors: List[str] = []
    first_player_recorded = False
    for m, raw_player, is_json in matches:
        # Record the FIRST tag's raw player id for audit, even if the box is
        # later rejected (so player_N placeholders are still observable).
        if not first_player_recorded:
            first_player_recorded = True
            if raw_player and raw_player.isdigit():
                out["first_player"] = int(raw_player)

        try:
            raw_coords = [float(m.group(k)) for k in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError):
            errors.append("non_numeric_coords")
            continue
        coords = _resolve_coordinate_scale(
            raw_coords, coordinate_mode, image_width, image_height, is_json
        )
        if coords is None:
            errors.append("out_of_range")
            continue
        x1, y1, x2, y2 = coords
        if not (x1 < x2 and y1 < y2):
            errors.append("bad_ordering")
            continue
        # A wrong player id loses the player-id credit but keeps the box's IoU.
        player_val = None
        if raw_player is not None:
            if not raw_player.isdigit():
                errors.append("invalid_player_id")
            else:
                player_val = int(raw_player)
                if valid_set is not None and player_val not in valid_set:
                    errors.append("invalid_player_id")
        out["players"].append(player_val)
        out["boxes_normalized"].append([x1, y1, x2, y2])
        if image_width and image_height:
            out["boxes_pixel"].append([
                x1 * image_width, y1 * image_height,
                x2 * image_width, y2 * image_height,
            ])

    out["num_valid"] = len(out["boxes_normalized"])
    out["bbox_valid"] = out["num_valid"] > 0
    out["errors"] = errors
    if not out["bbox_valid"]:
        out["bbox_parse_error"] = errors[0] if errors else "no_valid_bbox"
    return out


def compute_grounding_match(
    gold_boxes: List[List[float]],
    predicted_boxes: List[List[float]],
    match_mode: str = "recall",
    iou_threshold: float = 0.0,
    box_format: str = "xyxy",
) -> Dict[str, Any]:
    """Grounding overlap score with an optional precision term.

    ``recall`` (mean over gold of best IoU) cannot punish extra boxes, so
    covering the image with boxes is strictly dominant. ``f1`` matches boxes
    1-to-1 (greedy, highest IoU first), so every unmatched predicted box costs.
    """
    result = compute_grounding_iou(gold_boxes, predicted_boxes, box_format)
    result["match_mode"] = match_mode
    result["iou_threshold"] = iou_threshold
    if match_mode == "recall" and iou_threshold > 0 and result.get("per_box_ious"):
        kept = [i if i >= iou_threshold else 0.0 for i in result["per_box_ious"]]
        result["grounding_iou"] = sum(kept) / len(kept)
    if result.get("error") and match_mode == "recall":
        return result

    gold = [b for b in (_normalize_box(g, box_format) for g in gold_boxes) if b]
    pred = [b for b in (_normalize_box(p, box_format) for p in predicted_boxes) if b]
    if not gold or not pred:
        result.update({"grounding_recall": 0.0, "grounding_precision": 0.0, "grounding_f1": 0.0})
        if match_mode == "f1":
            result["grounding_iou"] = 0.0
        return result

    pairs = sorted(
        ((box_iou(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        key=lambda t: -t[0],
    )
    used_gold: set = set()
    used_pred: set = set()
    matched: List[float] = []
    for iou, gi, pi in pairs:
        if gi in used_gold or pi in used_pred:
            continue
        if iou < iou_threshold or iou <= 0.0:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matched.append(iou)

    total = sum(matched)
    recall = total / len(gold)
    precision = total / len(pred)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    result.update({
        "grounding_recall": recall,
        "grounding_precision": precision,
        "grounding_f1": f1,
        "num_matched_boxes": len(matched),
    })
    if match_mode == "f1":
        result["grounding_iou"] = f1
        result["error"] = None if matched else result.get("error")
    return result


# ---------------------------------------------------------------------------
# Model-bbox grounding scorer. Offline scoring and the live GRPO reward both
# go through here so they cannot drift apart.
# ---------------------------------------------------------------------------

def score_model_bbox_grounding(
    completion: Optional[str],
    gold_boxes_pixel: Optional[List[List[float]]],
    image_width: int = 320,
    image_height: int = 240,
    valid_player_ids: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score evidence boxes against gold boxes.

    reward = format_credit + player_id_credit + iou_credit * overlap, credits
    from ``config``, overlap = recall or box-F1 per ``match_mode``. A missing or
    invalid box scores 0.
    """
    from open_r1.self_evolve.reward_config import DEFAULT_COMPONENTS

    cfg = dict(DEFAULT_COMPONENTS["grounding"])
    cfg.update(config or {})
    parsed = parse_evidence_boxes(
        completion, image_width, image_height,
        valid_player_ids=valid_player_ids,
        coordinate_mode=cfg["coordinate_mode"],
    )

    fields: Dict[str, Any] = {
        "bbox_present": parsed["num_found"] > 0,
        "bbox_valid": bool(parsed["bbox_valid"]),
        "bbox_invalid_reason": None,
        "bbox_player_id": parsed.get("first_player"),
        "bbox_player_id_valid": False,
        "predicted_bbox_norm": parsed["boxes_normalized"] or None,
        "predicted_bbox_pixel": parsed["boxes_pixel"] or None,
        "predicted_players": parsed["players"] or None,
        "reference_bbox_norm": None,
        "metadata_bbox_iou": None,
        "grounding_reward": 0.0,
        "grounding_reward_components": {"format": 0.0, "player_id": 0.0, "iou": 0.0},
        "grounding_reward_source": "missing_model_bbox",
        "num_bbox_found": parsed["num_found"],
        "num_bbox_valid": parsed["num_valid"],
    }

    # Reference boxes in normalized coords (for audit) when image size is known.
    if gold_boxes_pixel and image_width and image_height:
        fields["reference_bbox_norm"] = [
            [b[0] / image_width, b[1] / image_height,
             b[2] / image_width, b[3] / image_height]
            for b in gold_boxes_pixel if isinstance(b, (list, tuple)) and len(b) == 4
        ] or None

    # player id validity (independent of coord validity, for stats)
    pid = parsed.get("first_player")
    fields["bbox_player_id_valid"] = pid is not None and (
        valid_player_ids is None or int(pid) in set(valid_player_ids)
    )

    if not fields["bbox_present"]:
        fields["grounding_reward_source"] = "missing_model_bbox"
        fields["bbox_invalid_reason"] = parsed.get("bbox_parse_error")  # "no_bbox_tag"/None
        if fields["bbox_invalid_reason"] in (None, "no_bbox_tag"):
            fields["bbox_invalid_reason"] = None
        return fields

    if not fields["bbox_valid"]:
        fields["grounding_reward_source"] = "invalid_model_bbox"
        fields["bbox_invalid_reason"] = parsed.get("bbox_parse_error") or "no_valid_bbox"
        return fields

    components = {
        "format": float(cfg["format_credit"]),
        "player_id": float(cfg["player_id_credit"]) if fields["bbox_player_id_valid"] else 0.0,
        "iou": 0.0,
    }
    fields["bbox_invalid_reason"] = "invalid_player_id" if "invalid_player_id" in parsed["errors"] else None
    fields["grounding_reward_components"] = components
    fields["grounding_reward"] = components["format"] + components["player_id"]

    # Valid model bbox in pixels.
    pred_pixel = parsed["boxes_pixel"] or None
    if not (pred_pixel and gold_boxes_pixel):
        fields["grounding_reward_source"] = "model_bbox_no_reference"
        return fields

    iou_result = compute_grounding_match(
        gold_boxes=gold_boxes_pixel,
        predicted_boxes=pred_pixel,
        match_mode=cfg["match_mode"],
        iou_threshold=float(cfg["iou_threshold"]),
        box_format="xyxy",
    )
    iou = float(iou_result["grounding_iou"])
    components["iou"] = float(cfg["iou_credit"]) * iou
    fields.update({
        "metadata_bbox_iou": iou,
        "grounding_reward": min(1.0, sum(components.values())),
        "grounding_reward_source": "model_bbox_iou",
        "per_box_ious": iou_result.get("per_box_ious", []),
        "grounding_recall": iou_result.get("grounding_recall"),
        "grounding_precision": iou_result.get("grounding_precision"),
        "grounding_match_mode": cfg["match_mode"],
    })
    return fields


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _is_valid_box(box: Any) -> bool:
    """Return True if *box* looks like [x1, y1, x2, y2] or [x, y, w, h]."""
    if not isinstance(box, (list, tuple)):
        return False
    if len(box) != 4:
        return False
    try:
        return all(isinstance(float(v), (int, float)) for v in box)
    except (TypeError, ValueError):
        return False


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = (float(v) for v in box)
    return [x, y, x + w, y + h]


def _normalize_box(box: Any, box_format: str = "xyxy") -> Optional[List[float]]:
    """Return *box* as [x1, y1, x2, y2], or None if invalid."""
    if not _is_valid_box(box):
        return None
    box_f = [float(v) for v in box]  # type: ignore[arg-type]
    if box_format == "xywh":
        return xywh_to_xyxy(box_f)
    return box_f


# ---------------------------------------------------------------------------
# Box IoU
# ---------------------------------------------------------------------------

def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-Union for two boxes in xyxy format.

    Returns 0.0 for invalid inputs or non-overlapping boxes.
    """
    try:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
    except (IndexError, TypeError, ValueError):
        return 0.0

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


# ---------------------------------------------------------------------------
# Multi-box grounding
# ---------------------------------------------------------------------------

def compute_grounding_iou(
    gold_boxes: List[List[float]],
    predicted_boxes: List[List[float]],
    box_format: str = "xyxy",
) -> Dict[str, Any]:
    """Compute multi-box grounding IoU.

    For each gold box we take the **maximum** IoU against any predicted box,
    then average across all gold boxes.

    Returns a dict with ``grounding_iou``, ``per_box_ious``, and any errors.
    """
    result: Dict[str, Any] = {
        "grounding_iou": 0.0,
        "per_box_ious": [],
        "box_format": box_format,
        "num_gold_boxes": len(gold_boxes),
        "num_predicted_boxes": len(predicted_boxes),
        "error": None,
    }

    if not gold_boxes:
        result["error"] = "no_gold_boxes"
        return result

    if not predicted_boxes:
        result["error"] = "no_predicted_boxes"
        return result

    gold_normalized = []
    for i, gb in enumerate(gold_boxes):
        nb = _normalize_box(gb, box_format)
        if nb is None:
            result["error"] = f"invalid_gold_box[{i}]: {gb}"
            return result
        gold_normalized.append(nb)

    pred_normalized = []
    for i, pb in enumerate(predicted_boxes):
        nb = _normalize_box(pb, box_format)
        if nb is None:
            # Skip invalid predicted boxes rather than failing entirely
            continue
        pred_normalized.append(nb)

    if not pred_normalized:
        result["error"] = "all_predicted_boxes_invalid"
        return result

    per_box_ious: List[float] = []
    for gb in gold_normalized:
        best_iou = max(box_iou(gb, pb) for pb in pred_normalized)
        per_box_ious.append(best_iou)

    result["per_box_ious"] = per_box_ious
    result["grounding_iou"] = sum(per_box_ious) / len(per_box_ious)
    return result


# ---------------------------------------------------------------------------
# Grounding score resolver (used by rewards.py)
# ---------------------------------------------------------------------------

def resolve_grounding_score(
    gold_evidence_boxes: Optional[List[List[float]]] = None,
    predicted_evidence_boxes: Optional[List[List[float]]] = None,
    box_format: str = "xyxy",
    grounding_score: Optional[float] = None,
    grounding_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve grounding score with IoU priority.

    Priority:
    1. If gold_boxes AND predicted_boxes → compute IoU.
    2. Else if grounding_score is not None → clamp it.
    3. Else → 0.0.

    Returns a dict suitable for ``reward_details["grounding"]``.
    """
    if gold_evidence_boxes is not None and predicted_evidence_boxes is not None:
        iou_result = compute_grounding_iou(
            gold_boxes=gold_evidence_boxes,
            predicted_boxes=predicted_evidence_boxes,
            box_format=box_format,
        )
        return {
            "grounding_source": "iou_boxes",
            "grounding_iou": iou_result["grounding_iou"],
            "gold_evidence_boxes": gold_evidence_boxes,
            "predicted_evidence_boxes": predicted_evidence_boxes,
            "box_format": box_format,
            "grounding_error": iou_result.get("error"),
            "per_box_ious": iou_result.get("per_box_ious", []),
            "grounding_score": iou_result["grounding_iou"],
        }

    if grounding_score is not None:
        clamped = max(0.0, min(1.0, float(grounding_score)))
        return {
            "grounding_source": grounding_source or "external_score",
            "grounding_iou": None,
            "gold_evidence_boxes": gold_evidence_boxes,
            "predicted_evidence_boxes": predicted_evidence_boxes,
            "box_format": box_format,
            "grounding_error": None,
            "grounding_score": clamped,
        }

    return {
        "grounding_source": "missing_grounding",
        "grounding_iou": None,
        "gold_evidence_boxes": None,
        "predicted_evidence_boxes": None,
        "box_format": box_format,
        "grounding_error": "no_grounding_data",
        "grounding_score": 0.0,
    }
