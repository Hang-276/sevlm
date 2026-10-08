
from functools import wraps
import inspect
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_modeling
from transformers.utils import logging
import torch
from typing import Tuple, Optional

logger = logging.get_logger(__name__)
Qwen2_5_VLVisionFlashAttention2 = getattr(qwen_modeling, "Qwen2_5_VLVisionFlashAttention2", None)
apply_rotary_pos_emb_flashatt = getattr(qwen_modeling, "apply_rotary_pos_emb_flashatt", None)
flash_attn_varlen_func = getattr(qwen_modeling, "flash_attn_varlen_func", None)

def qwen2_5vl_vision_flash_attn_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


def monkey_patch_qwen2_5vl_flash_attn():
    if Qwen2_5_VLVisionFlashAttention2 is None:
        if hasattr(qwen_modeling, "Qwen2_5_VLVisionAttention") and hasattr(qwen_modeling, "apply_rotary_pos_emb_vision"):
            # Modern attention already computes vision rotary embeddings in fp32.
            return False
        raise RuntimeError("Unsupported Qwen2.5-VL vision attention implementation")
    if apply_rotary_pos_emb_flashatt is None or flash_attn_varlen_func is None:
        raise RuntimeError("Legacy Qwen2.5-VL Flash Attention dependencies are unavailable")
    Qwen2_5_VLVisionFlashAttention2.forward = qwen2_5vl_vision_flash_attn_forward
    return True


# ----------------------- Fix the process pending bug when using data mixture of image-text data and pure-text under deepseed zero3-----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from typing import List, Union
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
def qwen2_5vl_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            has_images_global = False
            if pixel_values is not None:
                has_images_local = torch.tensor(1, device=input_ids.device)
            else:
                has_images_local = torch.tensor(0, device=input_ids.device)
            # Use all_reduce to ensure all GPUs know if there are images to process
            torch.distributed.all_reduce(has_images_local, op=torch.distributed.ReduceOp.MAX)
            has_images_global = has_images_local.item() > 0

            # If there are image inputs globally, ensure all GPUs call the visual model
            if has_images_global:
                if pixel_values is not None:   
                    pixel_values = pixel_values.type(self.visual.dtype)
                    # Handle case where image_grid_thw is None
                    if image_grid_thw is None:
                        # Generate default grid_thw based on pixel_values shape if available
                        if len(pixel_values.shape) == 2:
                            # Assume square image grid for simplicity
                            num_patches = pixel_values.shape[0]
                            # Common patch grid sizes, defaulting to 1x2x2 if can't determine
                            if num_patches == 1176:  # 28x28 + special tokens
                                image_grid_thw = torch.tensor([[1, 28, 28]], device=pixel_values.device)
                            elif num_patches == 576:  # 24x24
                                image_grid_thw = torch.tensor([[1, 24, 24]], device=pixel_values.device)
                            else:
                                # Default fallback
                                image_grid_thw = torch.tensor([[1, 2, 2]], device=pixel_values.device)
                        else:
                            # Default fallback for unknown shapes
                            image_grid_thw = torch.tensor([[1, 2, 2]], device=pixel_values.device)
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )
                    
                    mask = input_ids == self.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)
                    
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                else:
                    with torch.no_grad():
                        # Create a dummy image data for triggering parameter synchronization
                        dummy_pixel_values = torch.zeros((4, 1176), device=input_ids.device, dtype=self.visual.dtype)
                        dummy_grid_thw = torch.tensor([[1, 2, 2]], device=input_ids.device)
                        _ = self.visual(dummy_pixel_values, grid_thw=dummy_grid_thw)

            # Currently, video processing is not handled.
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                # Handle case where video_grid_thw is None
                if video_grid_thw is None:
                    # Generate default grid_thw for video
                    video_grid_thw = torch.tensor([[1, 2, 2]], device=pixel_values_videos.device)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

def monkey_patch_qwen2_5vl_forward():
    if hasattr(qwen_modeling, "Qwen2_5_VLTextModel"):
        model_class = qwen_modeling.Qwen2_5_VLModel
        original = model_class.forward
        if getattr(original, "_sevlm_modality_guard", False):
            return False
        signature = inspect.signature(original)

        @wraps(original)
        def homogeneous_forward(self, *args, **kwargs):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                values = signature.bind_partial(self, *args, **kwargs).arguments
                source = values.get("input_ids")
                if source is None:
                    source = values.get("inputs_embeds")
                device = source.device if source is not None else next(self.parameters()).device
                modalities = torch.tensor(
                    [values.get("pixel_values") is not None, values.get("pixel_values_videos") is not None],
                    device=device, dtype=torch.long,
                )
                torch.distributed.all_reduce(modalities, op=torch.distributed.ReduceOp.SUM)
                world = torch.distributed.get_world_size()
                if ((modalities != 0) & (modalities != world)).any():
                    raise RuntimeError(
                        "Qwen2.5-VL sharded training requires identical image/video presence across ranks per microbatch; "
                        "mixed image/text ranks are unsupported on this Transformers architecture"
                    )
            return original(self, *args, **kwargs)

        homogeneous_forward._sevlm_modality_guard = True
        model_class.forward = homogeneous_forward
        return True
    Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_forward
    return True

# ----------------------- Set the Weights only as False in torch.load (In Pytorch 2.6, this is default as True)-----------------------
def weigths_only_load(self, path: str, map_location=None):
    logger.info(f"[Torch] Loading checkpoint from {path}...")
    partition = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"[Torch] Loaded checkpoint from {path}.")
    return partition

def monkey_patch_torch_load():
    try:
        from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
    except ModuleNotFoundError as exc:
        if exc.name != "deepspeed":
            raise
        return False
    TorchCheckpointEngine.load = weigths_only_load
    return True


# ----------------------- DeepSpeed ZeRO-3 bf16 non-finite-grad skip -----------------------
# Root cause of the intermittent "loss finite but grad_norm=nan -> poisoned weights ->
# next rollout crashes" failure under the DeepSpeed backend:
#
# DeepSpeedZeroOptimizer_Stage3._overflow_check_and_loss_scale_update() only calls
# check_overflow() when self.dtype == torch.float16. Under bf16 (our full-param GRPO)
# it is skipped entirely, so self.overflow stays False and step()'s overflow early-return
# never fires -> a nan/inf gradient flows straight into unscale_and_clip_grads +
# _optimizer_step and writes nan weights. bf16 has no dynamic loss-scaler, so nothing
# else catches it. (fp16 gets this for free via the loss-scaler overflow path.)
#
# The trainer-level guards do NOT cover this path: GRPOTrainer.training_step short-circuits
# to super() for DeepSpeed (grad lives in the engine, not on p.grad), and compute_loss's
# A-2 gate only checks the *loss* (finite here). So the only correct interception point is
# inside the engine's own overflow check.
#
# DeepSpeed already implements a correct, cross-rank-synchronized inf/nan detector
# (has_overflow(): all_reduce MAX over every rank). We only need to also run it for bf16.
# When it trips, DeepSpeed's existing overflow path zeroes/skips the step for all ranks in
# lock-step (no desync, no hang) -- exactly the "skip this one step, don't train it" behaviour
# we want. This makes bf16 ZeRO-3 behave like fp16 w.r.t. non-finite gradients.
def monkey_patch_deepspeed_bf16_nan_skip():
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    if getattr(DeepSpeedZeroOptimizer_Stage3, "_bf16_nan_skip_patched", False):
        return

    _orig = DeepSpeedZeroOptimizer_Stage3._overflow_check_and_loss_scale_update

    def _patched_overflow_check_and_loss_scale_update(self):
        # For fp16 the original already runs check_overflow via the loss-scaler path; only
        # bf16 needs us to force the (cheap, all-reduced) inf/nan grad check before step().
        if self.dtype != torch.float16:
            self.check_overflow()
            if self.overflow:
                # Match the fp16 branch's bookkeeping: update the (no-op for bf16) scale so
                # loss_scale stays consistent, then let step() take its overflow early-return.
                prev_scale = self.loss_scale
                self._update_scale(self.overflow)
                self._overflow_clean_up(prev_scale)
                try:
                    import deepspeed.comm as dist
                    if dist.get_rank() == 0:
                        print("[GRAD-SKIP][deepspeed-bf16] non-finite gradient detected "
                              "-> skipping optimizer step (grad_norm=nan)", flush=True)
                except Exception:
                    pass
                return True
            return False
        return _orig(self)

    DeepSpeedZeroOptimizer_Stage3._overflow_check_and_loss_scale_update = (
        _patched_overflow_check_and_loss_scale_update
    )
    DeepSpeedZeroOptimizer_Stage3._bf16_nan_skip_patched = True


