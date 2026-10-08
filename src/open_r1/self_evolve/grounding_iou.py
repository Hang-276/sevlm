"""Evidence-box parsing and IoU grounding.

Parses the boxes a solver emits, matches them against the gold evidence boxes
(recall or box-F1), and turns that into the grounding reward.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# Accepted XML spellings include body coordinates, explicit x1/y1/x2/y2,
# coords/bbox2d attributes, and the common x1="x1,y1,x2,y2" generation error.
# The alt attribute is never interpreted as evidence: it often contains copied
# examples or natural-language labels. Every explicit coordinate source in one
# tag must agree, so the parser cannot cherry-pick a convenient box.
_BBOX_TAG_RE = re.compile(r"<bbox(?=[\s>])(?P<attrs>[^>]*)>(?P<body>.*?)</bbox>", re.I | re.S)
_BBOX_OPEN_RE = re.compile(r"<bbox(?=[\s>])[^>]*>", re.I | re.S)
_ATTR_RE = re.compile(
    r"\s+(?P<name>[a-zA-Z_][\w-]*)\s*=\s*"
    r"(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s\"'>]+))",
    re.S,
)
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")

# Qwen2.5-VL's native syntax, in absolute pixels — a format the solver was
# pretrained on rather than one it has to learn from a single example.
_BBOX_2D_RE = re.compile(
    r"\{(?P<obj>[^{}]*?\"bbox_2d\"\s*:\s*\[\s*"
    r"(?P<x1>-?\d*\.?\d+)\s*,\s*(?P<y1>-?\d*\.?\d+)\s*,\s*"
    r"(?P<x2>-?\d*\.?\d+)\s*,\s*(?P<y2>-?\d*\.?\d+)\s*\][^{}]*?)\}",
    re.IGNORECASE | re.DOTALL,
)
_PLAYER_JSON_RE = re.compile(
    r'"player"\s*:\s*(?:"([^\"]*)"|([^,}\s]+))(?=\s*(?:[,}]|$))', re.IGNORECASE
)
_PLAYER_JSON_KEY_RE = re.compile(r"\"player\"\s*:", re.IGNORECASE)
_BBOX_JSON_KEY_RE = re.compile(r"\"bbox_2d\"\s*:", re.IGNORECASE)


def _player_id(value: Optional[str]) -> Optional[int]:
    if not value or len(value) > 10 or not value.isascii() or not value.isdecimal():
        return None
    player = int(value)
    return player if 1 <= player <= 8 else None


def _allowed_players(values: Optional[List[int]]) -> Optional[set[int]]:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or any(type(value) is not int or not 1 <= value <= 8
                                                  for value in values):
        return set()
    return set(values)


def _coordinate_tuple(value: str) -> Optional[List[float]]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    elif value.startswith("[") or value.endswith("]"):
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or any(not _NUMBER_RE.fullmatch(part) for part in parts):
        return None
    coords = [float(part) for part in parts]
    return coords if all(math.isfinite(coord) for coord in coords) else None


def _parse_bbox_tag(attrs: str, body: str) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    if attrs.rstrip().endswith("/"):
        return None, None, "self_closing_bbox"
    parsed_attrs: Dict[str, List[str]] = {}
    offset = 0
    while offset < len(attrs):
        if not attrs[offset:].strip():
            break
        match = _ATTR_RE.match(attrs, offset)
        if match is None:
            return None, None, "malformed_attributes"
        name = match.group("name").lower()
        parsed_attrs.setdefault(name, []).append(
            next(v for v in (match.group("double"), match.group("single"), match.group("bare")) if v is not None)
        )
        offset = match.end()
    players = parsed_attrs.get("player", [])
    if len(players) > 1:
        return None, None, "conflicting_player_ids"
    player = players[0] if players else None

    candidates: List[List[float]] = []
    body_coords = _coordinate_tuple(body)
    if body_coords is not None:
        candidates.append(body_coords)
    elif body.strip().startswith("[") and body.count(",") >= 3:
        return None, player, "malformed_coords"

    for name in ("coords", "bbox2d", "bbox_2d", "bbox-2d"):
        for value in parsed_attrs.get(name, []):
            coords = _coordinate_tuple(value)
            if coords is None:
                return None, player, "malformed_coords"
            candidates.append(coords)

    xyxy = [parsed_attrs.get(name, []) for name in ("x1", "y1", "x2", "y2")]
    if any(xyxy):
        if any(len(values) > 1 for values in xyxy):
            return None, player, "conflicting_coords"
        if all(xyxy):
            values = [value[0] for value in xyxy]
            if any(not _NUMBER_RE.fullmatch(value.strip()) for value in values):
                return None, player, "malformed_coords"
            candidates.append([float(value) for value in values])
        elif xyxy[0] and not any(xyxy[1:]):
            coords = _coordinate_tuple(xyxy[0][0])
            if coords is None:
                return None, player, "incomplete_coords"
            candidates.append(coords)
        else:
            return None, player, "incomplete_coords"

    if not candidates:
        return None, player, "missing_coords"
    if any(any(abs(a - b) > 1e-6 for a, b in zip(candidates[0], candidate)) for candidate in candidates[1:]):
        return None, player, "conflicting_coords"
    return candidates[0], player, None


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
    ``valid_player_ids`` identifies the intended image. The legacy scorer can
    still use a box with the wrong id for coordinate learning; strict reward
    configs require every valid box to name the single gold player for IoU.

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

    # (position, raw coordinates, player, native JSON, parse error)
    matches: List[Tuple[int, Optional[List[float]], Optional[str], bool, Optional[str]]] = []
    tags = list(_BBOX_TAG_RE.finditer(completion))
    for match in tags:
        coords, player, error = _parse_bbox_tag(match.group("attrs"), match.group("body"))
        matches.append((match.start(), coords, player, False, error))
    for m in _BBOX_2D_RE.finditer(completion):
        raw_obj = m.group("obj") or ""
        pm = _PLAYER_JSON_RE.search(raw_obj)
        coords = [float(m.group(k)) for k in ("x1", "y1", "x2", "y2")]
        # Duplicate keys are ambiguous and a normal JSON loader keeps the
        # last value. Never reward a convenient first player or bbox key.
        parse_error = (
            "conflicting_player_ids" if len(_PLAYER_JSON_KEY_RE.findall(raw_obj)) > 1
            else "conflicting_coords" if len(_BBOX_JSON_KEY_RE.findall(raw_obj)) > 1
            else None
        )
        player = next((value for value in pm.groups() if value is not None), None) if pm else None
        matches.append((m.start(), coords, player, True, parse_error))
    # A malformed/unclosed bbox still counts as an attempted prediction. This
    # prevents an offline external-score fallback from paying for a bad tag.
    dangling = max(0, len(_BBOX_OPEN_RE.findall(completion)) - len(tags))
    matches.extend((len(completion), None, None, False, "unclosed_bbox") for _ in range(dangling))
    matches.sort(key=lambda item: item[0])

    out["num_found"] = len(matches)
    if not matches:
        out["bbox_parse_error"] = "no_bbox_tag"
        return out

    valid_set = _allowed_players(valid_player_ids)
    if valid_set == set():
        out["bbox_parse_error"] = "invalid_player_metadata"
        out["errors"] = ["invalid_player_metadata"]
        return out

    errors: List[str] = []
    first_player_recorded = False
    for _, raw_coords, raw_player, is_json, parse_error in matches:
        # Record the FIRST tag's raw player id for audit, even if the box is
        # later rejected (so player_N placeholders are still observable).
        if not first_player_recorded:
            first_player_recorded = True
            out["first_player"] = _player_id(raw_player)

        if parse_error is not None or raw_coords is None:
            errors.append(parse_error or "non_numeric_coords")
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
        # Keep a well-formed box for audit even if its player id is wrong. The
        # scorer decides whether strict mode zeroes its IoU contribution.
        player_val = None
        if raw_player is not None:
            player_val = _player_id(raw_player)
            if player_val is None:
                errors.append("invalid_player_id")
            else:
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

    # Player credit requires every scored box to name an allowed player. The
    # first raw tag is still logged for audit, but a valid-looking dummy tag
    # cannot lend its player id to a different box.
    pid = parsed.get("first_player")
    allowed = _allowed_players(valid_player_ids)
    valid_players = parsed["players"]
    fields["bbox_player_id_valid"] = bool(valid_players) and all(
        player is not None and (allowed is None or player in allowed)
        for player in valid_players
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
    if cfg["match_mode"] == "f1" and parsed["num_found"] > parsed["num_valid"]:
        # Invalid or unclosed tags are extra predictions, not free omissions.
        # The weighted IoU-F1 denominator is gold_count + pred_count.
        iou *= (len(gold_boxes_pixel) + parsed["num_valid"]) / (
            len(gold_boxes_pixel) + parsed["num_found"]
        )
    if cfg.get("require_correct_player_for_iou") and (
        allowed is None or len(allowed) != 1 or not fields["bbox_player_id_valid"]
    ):
        iou = 0.0
        fields["bbox_invalid_reason"] = "invalid_player_id"
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
