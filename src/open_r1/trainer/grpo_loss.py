"""GRPO loss reduction."""

from typing import Optional

import torch


def reduce_grpo_loss(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    loss_type: str = "grpo",
    max_completion_length: Optional[int] = None,
    loss_normalization_length: Optional[int] = None,
) -> torch.Tensor:
    """dr_grpo: sum(loss * mask) / (B * C), with fixed C (default: generation budget).

    Keep zero-advantage/masked rows in B. DDP/GA require equal sampled batch
    sizes; Trainer/Accelerate handles GA scaling. Callers guard non-finite loss.
    """
    if loss_type not in {"grpo", "dr_grpo"}:
        raise ValueError(f"Unsupported GRPO loss_type: {loss_type!r}")
    if loss_normalization_length is not None and loss_type != "dr_grpo":
        raise ValueError("loss_normalization_length is only supported with loss_type=dr_grpo")
    if per_token_loss.ndim != 2 or completion_mask.shape != per_token_loss.shape:
        raise ValueError("GRPO token loss and completion mask must have the same [batch, tokens] shape")
    if per_token_loss.shape[0] == 0:
        raise ValueError("GRPO loss requires a non-empty sampled batch")
    masked_loss = per_token_loss * completion_mask
    if loss_type == "grpo":
        return (masked_loss.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
    if not isinstance(max_completion_length, int) or isinstance(max_completion_length, bool) or max_completion_length <= 0:
        raise ValueError("dr_grpo requires a positive integer max_completion_length")
    if per_token_loss.shape[1] > max_completion_length:
        raise ValueError("Completion token width exceeds the fixed dr_grpo generation budget")
    normalization_length = max_completion_length if loss_normalization_length is None else loss_normalization_length
    if not isinstance(normalization_length, int) or isinstance(normalization_length, bool) or normalization_length <= 0:
        raise ValueError("loss_normalization_length must be a positive integer")
    return masked_loss.sum() / (per_token_loss.shape[0] * normalization_length)
