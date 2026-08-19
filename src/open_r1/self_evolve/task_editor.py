"""Task variants made by reverting part of a scene's changes.

The two renders share scene and camera, so pasting an object's original crop
back into the modified image reverts it exactly and the gold count drops by its
attribute-change count. K replaced objects therefore give 2^K-1 variants with
computed labels, which makes the label distribution a free variable and puts two
variants one object apart into a counterfactual pair.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from open_r1.self_evolve.task_difficulty import ATTRS

# Paste region around an object centre, in pixels. Matches the gold-box radius
# so a reverted object's evidence box disappears with its pixels.
DEFAULT_PATCH_RADIUS = 48


def changed_attribute_count(replaced_object: Dict[str, Any]) -> int:
    orig = replaced_object.get("original", {}) or {}
    repl = replaced_object.get("replacement", {}) or {}
    return sum(1 for a in ATTRS if orig.get(a) != repl.get(a))


def _centre(replaced_object: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for side in ("replacement", "original"):
        coords = (replaced_object.get(side, {}) or {}).get("pixel_coords")
        if coords and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
    return None


def _patch_box(
    replaced_object: Dict[str, Any],
    radius: int,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Union of the original and replacement footprints, clamped to the image."""
    boxes = []
    for side in ("original", "replacement"):
        coords = (replaced_object.get(side, {}) or {}).get("pixel_coords")
        if coords and len(coords) >= 2:
            cx, cy = float(coords[0]), float(coords[1])
            boxes.append((cx - radius, cy - radius, cx + radius, cy + radius))
    if not boxes:
        return None
    x1 = max(0, int(min(b[0] for b in boxes)))
    y1 = max(0, int(min(b[1] for b in boxes)))
    x2 = min(width, int(max(b[2] for b in boxes)))
    y2 = min(height, int(max(b[3] for b in boxes)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def gold_boxes_for(
    replaced_objects: Sequence[Dict[str, Any]],
    keep: Iterable[int],
    radius: int = DEFAULT_PATCH_RADIUS,
    image_width: int = 320,
    image_height: int = 240,
) -> List[List[float]]:
    """Evidence boxes for the objects a variant keeps changed."""
    keep = set(keep)
    boxes: List[List[float]] = []
    for i, ro in enumerate(replaced_objects):
        if i not in keep:
            continue
        centre = _centre(ro)
        if centre is None:
            continue
        cx, cy = centre
        boxes.append([
            max(0.0, cx - radius), max(0.0, cy - radius),
            min(float(image_width), cx + radius), min(float(image_height), cy + radius),
        ])
    return boxes


def variant_id(base_name: str, keep: Sequence[int]) -> str:
    return f"{base_name}__keep{'-'.join(str(i) for i in sorted(keep))}"


def build_variant(
    base_name: str,
    scene: Dict[str, Any],
    keep: Sequence[int],
    images_dir: Path,
    cache_dir: Path,
    radius: int = DEFAULT_PATCH_RADIUS,
) -> Optional[Dict[str, Any]]:
    """Splice the variant that keeps only ``keep`` changed; returns paths, label, boxes."""
    replaced = scene.get("modification", {}).get("replaced_objects") or []
    keep = sorted(set(int(i) for i in keep))
    if not keep or any(i >= len(replaced) or i < 0 for i in keep):
        return None

    original = images_dir / f"{base_name}_original.png"
    modified = images_dir / f"{base_name}_modified.png"
    if not (original.is_file() and modified.is_file()):
        return None

    num_attr_changes = sum(changed_attribute_count(replaced[i]) for i in keep)
    if num_attr_changes <= 0:
        return None

    # Reverting an object pastes a patch over its area; if that patch touches a
    # kept object's area, the kept change would be partially undone while the
    # gold label still claims it. Refuse such variants instead of corrupting.
    if _revert_overlaps_kept(replaced, keep, radius):
        return None

    # Keeping every object changed is the unedited scene; no splicing needed.
    if len(keep) == len(replaced):
        spy_image = modified
        spliced = False
    else:
        spy_image = cache_dir / f"{variant_id(base_name, keep)}_r{radius}.png"
        spliced = True
        if not spy_image.is_file():
            if not _splice(original, modified, replaced, keep, spy_image, radius):
                return None

    from PIL import Image

    with Image.open(modified) as im:
        width, height = im.size

    return {
        "variant_id": variant_id(base_name, keep),
        "base_name": base_name,
        "keep_indices": keep,
        "spy_image_path": str(spy_image),
        "civilian_image_path": str(original),
        "spliced": spliced,
        "num_changed_objects": len(keep),
        "num_attr_changes": num_attr_changes,
        "gold_boxes": gold_boxes_for(replaced, keep, radius, width, height),
        "image_width": width,
        "image_height": height,
    }


def _boxes_intersect(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _revert_overlaps_kept(replaced, keep, radius) -> bool:
    keep = set(keep)
    big = 10 ** 6
    reverted = [_patch_box(ro, radius, big, big)
                for i, ro in enumerate(replaced) if i not in keep]
    kept = [_patch_box(ro, radius, big, big)
            for i, ro in enumerate(replaced) if i in keep]
    return any(r and k and _boxes_intersect(r, k) for r in reverted for k in kept)


def _splice(
    original: Path,
    modified: Path,
    replaced: Sequence[Dict[str, Any]],
    keep: Sequence[int],
    out_path: Path,
    radius: int,
) -> bool:
    from PIL import Image

    keep = set(keep)
    try:
        with Image.open(modified) as mod_im, Image.open(original) as orig_im:
            canvas = mod_im.convert("RGB").copy()
            source = orig_im.convert("RGB")
            if source.size != canvas.size:
                return False
            pasted = 0
            for i, ro in enumerate(replaced):
                if i in keep:
                    continue
                box = _patch_box(ro, radius, canvas.width, canvas.height)
                if box is None:
                    # This object has to be reverted but there is nowhere to
                    # paste. Splicing the rest would leave it visibly changed
                    # while the gold count no longer counts it, so the image
                    # would contradict its own label. Refuse the variant.
                    return False
                canvas.paste(source.crop(box), box[:2])
                pasted += 1
            if pasted == 0:
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".tmp.png")
            canvas.save(tmp_path)
            tmp_path.replace(out_path)
        return True
    except Exception:
        return False


def enumerate_variants(
    scene: Dict[str, Any], max_variants: Optional[int] = None
) -> List[List[int]]:
    """All non-empty subsets of a scene's replaced objects, smallest first."""
    replaced = scene.get("modification", {}).get("replaced_objects") or []
    subsets: List[List[int]] = []
    for size in range(1, len(replaced) + 1):
        for combo in combinations(range(len(replaced)), size):
            subsets.append(list(combo))
            if max_variants and len(subsets) >= max_variants:
                return subsets
    return subsets


def counterfactual_pairs(subsets: Sequence[Sequence[int]]) -> List[Tuple[List[int], List[int]]]:
    """Subset pairs one object apart; a prior-following policy scores the same on both."""
    index = {tuple(sorted(s)): list(sorted(s)) for s in subsets}
    pairs: List[Tuple[List[int], List[int]]] = []
    for key, small in index.items():
        for bigger in index.values():
            if len(bigger) == len(small) + 1 and set(small) < set(bigger):
                pairs.append((small, bigger))
    return pairs
