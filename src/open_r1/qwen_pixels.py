"""Keep Qwen vision pixel limits consistent across slow, fast and vLLM paths."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Optional


def qwen_pixel_limits(processor: Any) -> dict[str, int]:
    """Return the limits actually used by the Qwen image preprocessor."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return {}
    size = getattr(image_processor, "size", None)
    if size is not None and not isinstance(size, MutableMapping):
        raise ValueError("Image processor size must be a mutable mapping")
    minimum = size.get("shortest_edge") if size is not None else getattr(image_processor, "min_pixels", None)
    maximum = size.get("longest_edge") if size is not None else getattr(image_processor, "max_pixels", None)
    return {
        name: value for name, value in (("min_pixels", minimum), ("max_pixels", maximum))
        if value is not None
    }


def configure_qwen_image_pixels(
    processor: Any, min_pixels: Optional[int], max_pixels: Optional[int],
) -> None:
    """Apply Qwen resolution overrides to both historical attributes and size.

    Qwen's fast image processor reads size when resizing. Updating only
    min_pixels/max_pixels silently changes the vLLM request but leaves
    the training image grid at the checkpoint's old resolution.
    """
    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if min_pixels is None and max_pixels is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Visual pixel limits require an image processor")
    current = qwen_pixel_limits(processor)
    effective_min = min_pixels if min_pixels is not None else current.get("min_pixels")
    effective_max = max_pixels if max_pixels is not None else current.get("max_pixels")
    if effective_min is not None and effective_max is not None and effective_min > effective_max:
        raise ValueError(
            f"Effective image pixel range is invalid: min={effective_min}, max={effective_max}"
        )
    size = getattr(image_processor, "size", None)
    if min_pixels is not None:
        image_processor.min_pixels = min_pixels
        if size is not None:
            size["shortest_edge"] = min_pixels
    if max_pixels is not None:
        image_processor.max_pixels = max_pixels
        if size is not None:
            size["longest_edge"] = max_pixels
