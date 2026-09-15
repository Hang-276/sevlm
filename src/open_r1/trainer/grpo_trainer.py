# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
# from trl import GRPOTrainer

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import re

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from .dynamic_dataset import DynamicIterableDataset, EpochAwareIterableDataset, CyclicDynamicDataset
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class CyclicRepeatSampler(Sampler):
    """
    Sampler that ensures the same samples are used within each num_iterations cycle.
    This is specifically designed for GRPO where we want to avoid regenerating images
    within the same optimization cycle.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):  
            Number of times to repeat each index (num_generations)
        batch_size (`int`):
            Number of unique indices per batch.
        cycle_length (`int`):
            Length of each cycle (num_iterations)
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int,
        cycle_length: int,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Pre-generate the sample order for the entire cycle
        self._generate_cycle_samples()

    def _generate_cycle_samples(self):
        """Generate samples for one complete cycle."""
        # Generate random indices for this cycle
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        
        # Group into batches
        batches = []
        for i in range(0, len(indexes), self.batch_size):
            batch = indexes[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
        
        # For each batch, create the cycle pattern
        self.cycle_samples = []
        for batch in batches:
            # Within each cycle, repeat the same batch cycle_length times
            for _ in range(self.cycle_length):
                for index in batch:
                    # Each index is repeated mini_repeat_count times
                    for _ in range(self.mini_repeat_count):
                        self.cycle_samples.append(index)

    def __iter__(self):
        # Regenerate samples for each epoch
        self._generate_cycle_samples()
        for sample in self.cycle_samples:
            yield sample

    def __len__(self) -> int:
        # Calculate total samples in one epoch
        num_complete_batches = self.num_samples // self.batch_size
        return num_complete_batches * self.batch_size * self.mini_repeat_count * self.cycle_length


class FiniteLogitsProcessor(LogitsProcessor):
    """A-3 rollout logits fallback.

    HF `generate` samples via `multinomial`, which triggers a CUDA device-side assert
    ("probability tensor contains either inf, nan or element < 0") and crashes every rank
    if the next-token logits contain nan/inf. When A-1/A-2 fail to catch a poisoned weight,
    this scrubs the logits right before sampling so a bad rollout degrades to a plausible
    token instead of taking down all 8 GPUs. nan -> very negative (unpicked), +inf -> a large
    finite value, -inf left as is. If a whole row is non-finite we reset it to uniform so
    multinomial always sees a valid distribution.

    It also hard-bans a set of special tokens (the image/video/vision markers) from being
    sampled: even after scrubbing, a uniform-reset row could sample e.g. `<|image_pad|>`
    (151655), which then makes the image-token count in prompt+completion exceed the number
    of vision features and raises "Image features and image tokens do not match". Setting
    those logits to -inf guarantees they can never appear in a completion."""

    def __init__(self, banned_token_ids=None):
        super().__init__()
        # Store as a plain list; converted to a tensor lazily on the scores' device.
        self.banned_token_ids = [int(t) for t in (banned_token_ids or []) if t is not None]
        self._banned_index = None  # cached LongTensor on the scores device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        non_finite = not torch.isfinite(scores).all()
        if non_finite:
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e9)
            # Guard against an all-(-1e9) row (e.g. originally all-nan) producing a degenerate softmax.
            bad_rows = ~torch.isfinite(scores).any(dim=-1) | (scores.max(dim=-1).values <= -1e9)
            if bad_rows.any():
                scores[bad_rows] = 0.0  # uniform logits -> uniform sampling for that row
        # Ban the special tokens unconditionally (applied after any uniform reset so the
        # reset can't re-open them). Cheap masked_fill on a tiny index tensor.
        if self.banned_token_ids:
            if self._banned_index is None or self._banned_index.device != scores.device:
                self._banned_index = torch.tensor(
                    self.banned_token_ids, dtype=torch.long, device=scores.device
                )
            scores[:, self._banned_index] = float("-inf")
        return scores


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        script_args: Optional[Any] = None,  # Add script_args parameter
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        self._rollout_model_id = model_id
        self._vllm_rollout = None
        self._vllm_generation_count = 0
        self.use_vllm = args.use_vllm
        if self.use_vllm and self.vlm_module.get_vlm_key() != "qwen":
            raise ValueError("GRPO vLLM rollouts currently support Qwen2/2.5-VL; use_vllm=False for other VLMs.")
        if self.use_vllm and args.vllm_device != "auto":
            raise ValueError("GRPO vLLM shares each rank's training GPU. Set vllm_device=auto.")
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            print("Applying LoRA...")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
        # Compute the number of trainable parameters and print the parameter that is trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)
        print(f"Total trainable parameters: {total_params}")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        # Both ZeRO-3 and FSDP shard the params, so create_reference_model()'s
        # deepcopy can't see full weights — reload the ref model from disk under
        # either sharded backend. (self.is_fsdp_enabled isn't set until
        # super().__init__(), so detect FSDP straight from args.fsdp here.)
        _fsdp_enabled = bool(getattr(args, "fsdp", None))
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled() or _fsdp_enabled:
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    # If we cannot find component in processing_class, return the processing_class itself
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Rollout special-token ban list (fed to FiniteLogitsProcessor). These vision
        # markers must never be *sampled* into a completion: a sampled `<|image_pad|>`
        # makes the image-token count in prompt+completion exceed the vision-feature
        # count and raises "Image features and image tokens do not match" (the exact
        # crash the nan-grad chain produced). Resolve by name via the tokenizer so it
        # works across Qwen2/2.5-VL; fall back to Qwen2.5-VL's known ids.
        self._rollout_banned_token_ids = self._resolve_banned_rollout_token_ids(processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=args.temperature,
            pad_token_id=pad_token_id,
        )
        self.dynamic_sampling = args.dynamic_sampling
        self.dynamic_sampling_std_threshold = args.dynamic_sampling_std_threshold
        self.dynamic_sampling_max_retries = max(0, args.dynamic_sampling_max_retries)
        self.dynamic_sampling_mode = getattr(args, "dynamic_sampling_mode", "batch_retry")
        self.scale_rewards = getattr(args, "scale_rewards", True)
        self.overlong_filtering = getattr(args, "overlong_filtering", False)
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        
        # Store script_args for later access
        self.script_args = script_args
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers
        )

        # Two-phase training control for CLEVR (moved after super().__init__ to access self.accelerator)
        # 'clue' - only train clue phase, 'decision' - only train decision phase, 'both' - train both phases
        self.clevr_training_phase = getattr(script_args, 'training_phase', 'decision')  # Default to decision phase only
        if self.accelerator.process_index == 0:
            print(f"[CLEVR] Training phase set to: {self.clevr_training_phase}")
            if self.clevr_training_phase == 'clue':
                print("[CLEVR] Only clue phase will be trained, decision phase will use torch.no_grad()")
            elif self.clevr_training_phase == 'decision':
                print("[CLEVR] Only decision phase will be trained, clue phase will use torch.no_grad()")
            elif self.clevr_training_phase == 'both':
                print("[CLEVR] Both phases will be trained")
            else:
                print(f"[CLEVR] Warning: Unknown training phase '{self.clevr_training_phase}', defaulting to 'both'")
                self.clevr_training_phase = 'both'

        # Check if the per-device-train/eval_batch_size * num processes can be divided by the number of generations
        # Skip this check for CLEVR spot-the-difference games since they use group-based training
        data_generator_type = getattr(script_args, 'data_generator_type', None) if script_args else None
        print(f"[DEBUG] data_generator_type: {data_generator_type}")
        is_clevr_spotdiff = data_generator_type == 'clevr_spotdiff'
        print(f"[DEBUG] is_clevr_spotdiff: {is_clevr_spotdiff}")
        
        # Check if using ZeRO Stage 3 for model parallel
        is_zero3_enabled = is_deepspeed_zero3_enabled()
        print(f"[DEBUG] ZeRO Stage 3 enabled: {is_zero3_enabled}")
        # FSDP shards params just like ZeRO-3 (data-parallel + sharded), so the
        # global-batch divisibility check should be skipped for it too.
        is_fsdp_enabled = bool(getattr(self, "is_fsdp_enabled", False))
        print(f"[DEBUG] FSDP enabled: {is_fsdp_enabled}")
        is_sharded_enabled = is_zero3_enabled or is_fsdp_enabled

        if not is_clevr_spotdiff and not is_sharded_enabled:
            num_processes = self.accelerator.num_processes
            global_batch_size = args.per_device_train_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                    f"batch size, the valid values for the number of generations are: {possible_values}."
                )
            if self.args.eval_strategy != "no":
                global_batch_size = args.per_device_eval_batch_size * num_processes
                possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
                if self.num_generations not in possible_values:
                    raise ValueError(
                        f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                        f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                        f"eval batch size, the valid values for the number of generations are: {possible_values}."
                    )
        elif is_clevr_spotdiff:
            # For CLEVR spot-the-difference games, log that we're using group-based training
            if self.accelerator.process_index == 0:
                print(f"CLEVR spot-the-difference detected: Using group-based training")
                if is_sharded_enabled:
                    _which = "ZeRO Stage 3" if is_zero3_enabled else "FSDP"
                    print(f"{_which} Model Parallel: {self.accelerator.num_processes} GPUs collaborating on each game")
                else:
                    print(f"Data Parallel: {self.accelerator.num_processes} GPUs processing separate games")
                print(f"Per-device batch size: {args.per_device_train_batch_size}")
                print(f"Number of generations: {self.num_generations}")
        elif is_sharded_enabled:
            # For ZeRO Stage 3 / FSDP model parallel, skip batch size checks
            if self.accelerator.process_index == 0:
                _which = "ZeRO Stage 3" if is_zero3_enabled else "FSDP"
                print(f"{_which} Model Parallel enabled: {self.accelerator.num_processes} GPUs collaborating")
                print(f"Per-device batch size: {args.per_device_train_batch_size}")
                print(f"Number of generations: {self.num_generations}")

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            if self._vllm_rollout is not None:
                self._vllm_rollout.close()
                self._vllm_rollout = None

    def _generate_with_vllm(self, model, prompts_text, images_by_prompt, prompt_ids, prompt_mask):
        from .vllm_rollout import ColocatedVLLMRollout, build_requests, completion_tensors

        if self._vllm_rollout is None:
            if self.is_fsdp_enabled and self.accelerator.state.fsdp_plugin.fsdp_version != 2:
                raise ValueError("GRPO vLLM weight synchronization requires FSDP2, not FSDP1.")
            image_processor = getattr(self.processing_class, "image_processor", None)
            processor_kwargs = {
                k: getattr(image_processor, k) for k in ("min_pixels", "max_pixels")
                if getattr(image_processor, k, None) is not None
            }
            engine_kwargs = dict(
                model=self._rollout_model_id,
                dtype=self.args.vllm_dtype,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                max_model_len=self.args.vllm_max_model_len or 32768,
                max_num_seqs=self.args.vllm_max_num_seqs,
                limit_mm_per_prompt={"image": self.args.vllm_max_images},
                mm_processor_kwargs=processor_kwargs,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                seed=self.args.seed + self.accelerator.process_index,
            )
            torch.cuda.empty_cache()
            self._vllm_rollout = ColocatedVLLMRollout(
                device_index=self.accelerator.device.index,
                engine_kwargs=engine_kwargs,
                log_path=os.path.join(self.args.output_dir, f"vllm_rank{self.accelerator.process_index}.log"),
                timeout=self.args.vllm_worker_timeout,
            )
        sampling = dict(
            n=1, max_tokens=self.max_completion_length,
            temperature=self.args.temperature, top_p=self.args.top_p,
            top_k=self.args.top_k if self.args.top_k is not None else -1,
            min_p=self.args.min_p or 0.0, repetition_penalty=self.args.repetition_penalty,
            seed=self.args.seed + self.accelerator.process_index * 1000003 + self._vllm_generation_count,
            banned_token_ids=self._rollout_banned_token_ids,
        )
        if self.args.vllm_guided_decoding_regex is not None:
            raise ValueError("Guided decoding is not supported by the synchronous GRPO rollout path.")
        self._vllm_generation_count += 1
        unwrapped = self.accelerator.unwrap_model(model)
        outputs = self._vllm_rollout.generate(
            unwrapped, self.state.global_step, build_requests(prompts_text, images_by_prompt), sampling,
        )
        expected = [ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, prompt_mask)]
        result = completion_tensors(outputs, expected, self.processing_class.pad_token_id, prompt_ids.device)
        self._metrics["rollout/vllm"].append(1.0)
        self._metrics["rollout/policy_step"].append(float(self.state.global_step))
        return result

    def _rollout_logits_processor(self) -> "LogitsProcessorList":
        """A fresh LogitsProcessorList carrying the A-3 finite-scrub + special-token
        ban, to attach to every rollout `generate` call (including the CLEVR
        two-phase clue/decision generations, which are the real training route and
        where the nan-grad → sampled <|image_pad|> crash surfaced). A new instance
        per call keeps the tiny cached device tensor local and thread-safe."""
        return LogitsProcessorList([
            FiniteLogitsProcessor(banned_token_ids=getattr(self, "_rollout_banned_token_ids", None))
        ])

    def _resolve_banned_rollout_token_ids(self, processing_class) -> list:
        """Token ids that FiniteLogitsProcessor forbids the rollout from sampling.

        Vision markers only: <|image_pad|>, <|video_pad|>, <|vision_start|>,
        <|vision_end|>. Resolve by name from the tokenizer so it tracks whatever
        vocab is actually loaded; fall back to Qwen2.5-VL's fixed ids if a name
        isn't found. Never bans eos/pad — those are legitimate completion tokens.
        """
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        # Qwen2.5-VL fixed ids, used as fallback if convert_tokens_to_ids misses.
        fallback = {
            "<|image_pad|>": 151655,
            "<|video_pad|>": 151656,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
        }
        unk_id = getattr(tokenizer, "unk_token_id", None)
        banned = set()
        for tok, fb in fallback.items():
            tid = None
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                try:
                    tid = convert(tok)
                except Exception:
                    tid = None
            # convert_tokens_to_ids returns unk_id for unknown tokens -> treat as miss.
            if tid is None or (unk_id is not None and tid == unk_id):
                tid = fb
            if tid is not None:
                banned.add(int(tid))
        banned_list = sorted(banned)
        try:
            if self.accelerator.process_index == 0:
                print(f"[ROLLOUT-BAN] special token ids banned from sampling: {banned_list}")
        except Exception:
            pass
        return banned_list

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to skip the optimizer update on a non-finite gradient (nan/inf).

        bf16 has no dynamic loss-scaler overflow-skip, so a single degenerate batch
        that produces a finite loss but a nan/inf gradient will otherwise write nan
        weights, which then poison every subsequent rollout. The parent's A-2 gate
        only checks the *loss* (finite here), so it misses this. We run the parent
        training_step (forward + accelerator.backward populates .grad), then inspect
        the gathered grads across all ranks; if any rank saw a non-finite grad we zero
        the grads on every rank (making this step a no-op) so the ranks stay in sync.

        Only active for FSDP/non-DeepSpeed backends: under DeepSpeed ZeRO-3 the
        optimizer step happens inside accelerator.backward, so .grad isn't inspectable
        here and the existing in-engine handling stands. For DeepSpeed we defer to the
        parent unchanged.
        """
        # DeepSpeed: leave the parent behaviour untouched (grad lives in the engine).
        if getattr(self, "is_deepspeed_enabled", False):
            return super().training_step(model, inputs, num_items_in_batch)

        loss = super().training_step(model, inputs, num_items_in_batch)

        # After super().training_step, backward has run and .grad is populated (the
        # optimizer.step() happens later in the Trainer loop, so we can still cancel).
        local_bad = torch.zeros(1, device=self.accelerator.device)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                local_bad += 1
                break
        # Sync the skip decision across ranks so all of them zero_grad together and
        # nobody desyncs by stepping while others skip.
        try:
            gathered = self.accelerator.reduce(local_bad, reduction="sum")
        except Exception:
            gathered = local_bad
        if gathered.item() > 0:
            # Zero every grad -> optimizer.step() becomes a no-op this step.
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            if self.accelerator.is_main_process:
                print(f"[GRAD-SKIP] step={self.state.global_step} non-finite gradient detected -> skipping weight update (grad_norm=nan)")
            # Return a zero loss so downstream logging shows the skip cleanly.
            return torch.zeros_like(loss) if isinstance(loss, torch.Tensor) else loss
        return loss

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            if getattr(model, "language_model", None) is not None:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False
            else:
                model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        
        # Process in smaller chunks to reduce memory usage without changing logic
        batch_size = logits.size(0)
        chunk_size = min(4, batch_size)  # Process at most 4 samples at a time to reduce memory
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_logits = logits[i:end_idx]
            chunk_input_ids = input_ids[i:end_idx]
            
            chunk_per_token_logps = []
            for logits_row, input_ids_row in zip(chunk_logits, chunk_input_ids):
                # fp32 log_softmax: bf16 logits (8-bit mantissa) lose precision here and can
                # produce nan/inf logprobs on extreme logits, which then blows up the KL exp
                # term and the backward grad_norm. Upcasting to fp32 is the standard fix.
                log_probs = logits_row.float().log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                chunk_per_token_logps.append(token_log_prob)
            
            if chunk_per_token_logps:
                chunk_result = torch.stack(chunk_per_token_logps)
                per_token_logps.append(chunk_result)
        
        # Concatenate all chunks or return original computation if chunking failed
        if per_token_logps:
            return torch.cat(per_token_logps, dim=0)
        else:
            # Fallback to original logic if chunking doesn't work
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.float().log_softmax(dim=-1)  # fp32 for numerical stability (see chunked path)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate a non-degenerate rollout batch with bounded DAPO-style resampling.

        All ranks retry together so groups that span devices remain aligned.
        The retry bound prevents a hard prompt from stalling training forever.
        """
        # 'mask_degenerate' neutralises degenerate groups during scoring, so
        # there is nothing to resample for.
        resample = self.dynamic_sampling and self.dynamic_sampling_mode == "batch_retry"
        retries = self.dynamic_sampling_max_retries if resample else 0
        for attempt in range(retries + 1):
            metric_lengths = {key: len(values) for key, values in self._metrics.items()}
            result = self._generate_and_score_completions_once(inputs, model)
            group_stds = result.pop("_group_reward_stds")
            degenerate = sum(std <= self.dynamic_sampling_std_threshold for std in group_stds)
            if not degenerate or attempt == retries:
                self._metrics["dynamic_sampling/retries"].append(float(attempt))
                self._metrics["dynamic_sampling/degenerate_groups"].append(float(degenerate))
                if degenerate and self.accelerator.process_index == 0:
                    print(f"[DYNAMIC SAMPLING] retry limit reached; keeping {degenerate} degenerate group(s)")
                pending = getattr(self, "_pending_rollout_dump", None)
                if pending is not None:
                    self._maybe_dump_step_rollouts(*pending)
                    self._pending_rollout_dump = None
                return result

            # Rejected rollouts must not pollute training metrics.
            for key in list(self._metrics):
                del self._metrics[key][metric_lengths.get(key, 0):]
            if self.accelerator.process_index == 0:
                print(f"[DYNAMIC SAMPLING] discarded batch with {degenerate} near-zero-std group(s); retry {attempt + 1}/{retries}")

    def _generate_and_score_completions_once(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        try:
            from open_r1.self_evolve.live_reward import set_training_step

            set_training_step(self.state.global_step)
        except Exception:
            pass
        prompts = [x["prompt"] for x in inputs]
        
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        # Handle both pre-loaded images and image paths
        images = []
        images_by_prompt = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            else:
                imgs = []

            prompt_images = []
            for img in imgs:
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)
                prompt_images.append(img)
            images_by_prompt.append(prompt_images)
                

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            # Qwen's HF processor expands image placeholders in this list in
            # place. vLLM must receive the original, unexpanded chat prompts.
            list(prompts_text),
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device first, then handle dtype
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        # Ensure dtypes match the model - particularly important for InternVL with bfloat16
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'input_ids' in prompt_inputs:
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(dtype=torch.long)  # Ensure correct dtype for input_ids
            if 'attention_mask' in prompt_inputs:
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(dtype=torch.long)  # Ensure correct dtype for attention_mask
        elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
            target_dtype = model.config.torch_dtype
            if isinstance(target_dtype, str):
                target_dtype = getattr(torch, target_dtype)
            if target_dtype is not None and 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=target_dtype)
        
        # Additional fallback: ensure pixel_values is at least bfloat16 for InternVL and on correct device
        if 'pixel_values' in prompt_inputs:
            if prompt_inputs['pixel_values'].dtype == torch.float32:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            # Ensure it's on the right device
            prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(device)
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions. A-3: attach FiniteLogitsProcessor so a poisoned weight (non-finite
        # logits) degrades the rollout instead of crashing all ranks in multinomial sampling.
        if self.use_vllm:
            prompt_length = prompt_ids.size(1)
            completion_ids, vllm_completion_mask, vllm_truncated = self._generate_with_vllm(
                model, prompts_text, images_by_prompt, prompt_ids, prompt_mask,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=self.generation_config,
                    logits_processor=self._rollout_logits_processor(),
                )
                prompt_length = prompt_ids.size(1)
                if not self.vlm_module.is_embeds_input():
                    prompt_completion_ids = generate_returned_result
                    prompt_ids = prompt_completion_ids[:, :prompt_length]
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                else:
                    completion_ids = generate_returned_result
                    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        if self.use_vllm:
            completion_mask = vllm_completion_mask

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:

                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        if self.scale_rewards:
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        else:
            # No std division: dividing rescales every group to unit spread, so
            # format noise in a saturated group gets full-strength gradient.
            advantages = rewards - mean_grouped_rewards
        if self.dynamic_sampling and self.dynamic_sampling_mode == "mask_degenerate":
            # Drop only the degenerate groups instead of the whole batch.
            degenerate_mask = (std_grouped_rewards <= self.dynamic_sampling_std_threshold)
            advantages = advantages * (~degenerate_mask).float()
            self._metrics["dynamic_sampling/masked_groups"].append(
                (degenerate_mask.view(-1, self.num_generations)[:, 0]).float().sum().item()
            )
        # A rollout cut off at max_completion_length has no </answer>, so its
        # near-zero reward reflects the length cap, not the reasoning.
        truncated_local = vllm_truncated if self.use_vllm else ~is_eos.any(dim=1)
        truncated = self.accelerator.gather(truncated_local.int()).bool()
        self._metrics["truncated_fraction"].append(truncated.float().mean().item())
        if self.overlong_filtering:
            advantages = advantages * (~truncated).float()
        # Scrub non-finite advantages at the source: a reward function returning nan/inf
        # (e.g. a judge timeout, a grounding divide-by-zero, a flaky live GPT-4o call) would
        # otherwise propagate straight into per_token_loss -> grad_norm=nan -> poisoned weights
        # -> next rollout crashes with "probability tensor contains inf/nan".
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        # std_grouped_rewards is already global at this point; no second gather.
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        # Per-dimension reward curves (answer / grounding / process / consistency /
        # budget). The training scalar `reward` above is the weighted sum; these
        # break it into its five raw components so wandb shows, e.g., the answer
        # accuracy trend on its own. Values come from the reward func breakdowns
        # (local slice), gathered across processes for a global per-step mean.
        try:
            from open_r1.self_evolve.live_reward import get_last_breakdowns

            _bd = get_last_breakdowns() or []
            if _bd:
                for _dim in ("answer", "grounding", "process", "consistency", "budget"):
                    _vals = [float(b.get(_dim, 0.0)) for b in _bd if isinstance(b, dict) and b.get(_dim) is not None]
                    if not _vals:
                        continue
                    _local_mean = torch.tensor(sum(_vals) / len(_vals), device=device, dtype=torch.float32)
                    _global_mean = self.accelerator.gather_for_metrics(_local_mean).mean().item()
                    self._metrics[f"reward/{_dim}"].append(_global_mean)
        except Exception as _e:
            print(f"[REWARD-DIM-METRICS] skipped: {_e}")

        # Per-step trajectory + reward dump (opt-in via SELF_EVOLVE_GRPO_DUMP_DIR).
        # Uses local-rank tensors: `prompts`/`completions` are local, `advantages`
        # is already sliced to this rank, and the five-dim breakdowns come from
        # the reward func aligned 1:1 with local completions.
        # Deferred: the retry wrapper dumps only the accepted attempt, so
        # discarded batches never pollute the per-step rollout dump.
        self._pending_rollout_dump = (prompts, completions, inputs, advantages)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "_group_reward_stds": std_grouped_rewards[::self.num_generations].tolist(),
        }

    def _maybe_dump_step_rollouts(self, prompts, completions, inputs, advantages):
        """Dump every rollout of this step (traj text + five-dim reward) to disk.

        Opt-in: only runs when SELF_EVOLVE_GRPO_DUMP_DIR is set. Layout is one
        directory per global step, one jsonl per rank inside it:

            <dump_dir>/step_000042/rank0.jsonl

        Each rank writes only its own local slice, so there is no cross-process
        gather and no write lock — a step's full batch is the union of its
        rankN.jsonl files. Failures here never interrupt training.
        """
        dump_dir = os.getenv("SELF_EVOLVE_GRPO_DUMP_DIR")
        if not dump_dir:
            return
        try:
            from open_r1.self_evolve.live_reward import get_last_breakdowns

            step = int(self.state.global_step)
            rank = int(self.accelerator.process_index)
            breakdowns = get_last_breakdowns() or []

            def _text(c):
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    return c[0].get("content", "")
                if isinstance(c, dict):
                    return c.get("content", "")
                return str(c or "")

            def _prompt_text(p):
                if isinstance(p, list):
                    return "\n".join(
                        f"{m.get('role', '')}: {m.get('content', '')}"
                        if isinstance(m, dict) else str(m)
                        for m in p
                    )
                return str(p or "")

            adv_list = advantages.detach().float().cpu().tolist() if hasattr(advantages, "detach") else list(advantages)

            step_dir = os.path.join(dump_dir, f"step_{step:06d}")
            os.makedirs(step_dir, exist_ok=True)
            out_path = os.path.join(step_dir, f"rank{rank}.jsonl")

            n = len(completions)
            with open(out_path, "a", encoding="utf-8") as fh:
                for i in range(n):
                    ex = inputs[i] if i < len(inputs) else {}
                    rec = {
                        "global_step": step,
                        "rank": rank,
                        "index_in_rank": i,
                        "problem": ex.get("problem") if isinstance(ex, dict) else None,
                        "solution": ex.get("solution") if isinstance(ex, dict) else None,
                        "image_path": ex.get("image_path") if isinstance(ex, dict) else None,
                        "prompt": _prompt_text(prompts[i]) if i < len(prompts) else None,
                        "completion": _text(completions[i]),
                        "advantage": adv_list[i] if i < len(adv_list) else None,
                        "reward_breakdown": breakdowns[i] if i < len(breakdowns) else None,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            # Dumping is best-effort observability; never let it crash training.
            print(f"[GRPO-DUMP] step={getattr(self.state, 'global_step', '?')} dump failed: {e}")

    def _handle_clevr_spotdiff_training(self, inputs: list[dict], model) -> dict:
        """
        Handle two-phase training for CLEVR spot-the-difference game:
        1. Clue phase: Generate clues for each player across multiple rounds (now participates in training)
        2. Decision phase: Each player makes decisions (continues to participate in training)
        Both phases get rewards after each complete round
        """
        # Import here to avoid circular imports
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        device = self.accelerator.device
        
        # Get training phase from script_args
        base_training_phase = getattr(self.script_args, 'training_phase', 'both')
        
        # Handle interactive training mode (step-based only)
        if base_training_phase == 'interactive':
            # Get configurable cycle length (in steps)
            interactive_cycle_length = getattr(self.script_args, 'interactive_cycle_length', 10)
            
            # Direct step-based cycling
            current_step = self.state.global_step
            total_cycle_length = interactive_cycle_length * 2  # Full cycle = decision + clue phases
            cycle_position = current_step % total_cycle_length
            
            if cycle_position < interactive_cycle_length:
                training_phase = 'decision'  # First half of cycle
            else:
                training_phase = 'clue'      # Second half of cycle
            
            if self.accelerator.process_index == 0:
                print(f"[INTERACTIVE MODE] Cycle length: {interactive_cycle_length} steps per phase")
                print(f"[INTERACTIVE MODE] Current step: {current_step}, Cycle position: {cycle_position}/{total_cycle_length}, Active phase: {training_phase}")
        else:
            training_phase = base_training_phase
        
        # Memory monitoring
        if hasattr(torch.cuda, 'memory_allocated') and self.accelerator.process_index == 0:
            memory_before = torch.cuda.memory_allocated(device) / 1024**3  # GB
            print(f"[MEMORY] Before CLEVR processing: {memory_before:.2f} GB")
            print(f"[TRAINING] Current training phase: {training_phase}")
            if base_training_phase == 'interactive':
                print(f"[TRAINING] Base mode: interactive, Active phase: {training_phase}, Global step: {self.state.global_step}")
        
        # DEBUG: Log batch information (DISABLED for cleaner output)
        # if self.accelerator.process_index == 0:
        #     print(f"[DEBUG] Device {self.accelerator.process_index}: Received {len(inputs)} samples in batch")
        #     print(f"[DEBUG] global_step: {self.state.global_step}, num_iterations: {self.num_iterations}")
        #     print(f"[DEBUG] gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")
        #     print(f"[DEBUG] Training phase: {training_phase}")
        
        # Check if we need to generate new data or use buffered ones
        step_key = self._step % self.args.gradient_accumulation_steps
        buffer_key = f"clevr_spotdiff_{step_key}"
        
        if self.state.global_step % self.num_iterations == 0:
            if self.accelerator.process_index == 0:
                if base_training_phase == 'interactive':
                    interactive_cycle_length = getattr(self.script_args, 'interactive_cycle_length', 10)
                    current_step = self.state.global_step
                    total_cycle_length = interactive_cycle_length * 2
                    cycle_position = current_step % total_cycle_length
                    print(f"=== CLEVR Interactive Training: Currently in {training_phase.upper()} Phase ===")
                    print(f"[DEBUG] Step {current_step}, Cycle position {cycle_position}/{total_cycle_length}, Active phase: {training_phase}")
                    print(f"[DEBUG] Cycle length: {interactive_cycle_length} steps per phase")
                else:
                    print(f"=== CLEVR Two-Phase Training: {training_phase.upper()} Phase ===")
                print(f"[DEBUG] Will process {len(inputs)} games")
                print(f"[DEBUG] This is a NEW game generation (global_step={self.state.global_step}, num_iterations={self.num_iterations})")
            
            # Collect all training samples from both phases
            all_training_samples = []
            all_phase_labels = []  # Track which phase each sample belongs to
            all_game_metadata = []
            
            # Process only the first game to avoid OOM (each device should only process one game)
            if len(inputs) > 1:
                if self.accelerator.process_index == 0:
                    print(f"[WARNING] Received {len(inputs)} samples, but will only process the first one to avoid OOM")
                inputs = [inputs[0]]  # Only process the first sample
            
            # Process each game
            for i, sample in enumerate(inputs):
                            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] Processing game {i+1}/{len(inputs)}")
                game_data = sample["game_data"]
                num_players = game_data["num_players"]
                num_rounds = getattr(self, '_clevr_num_rounds', 2)
                
                # Phase 1: Generate clues - use no_grad if not training clue phase
                if training_phase in ["clue", "both"]:
                    if self.accelerator.process_index == 0:
                        print(f"[TRAINING] Generating clue phase WITH gradients")
                    clue_samples, clue_responses = self._generate_clue_phase_for_training(
                        sample, model, num_rounds, requires_grad=True
                    )
                else:
                    if self.accelerator.process_index == 0:
                        print(f"[TRAINING] Generating clue phase WITHOUT gradients (inference only)")
                    with torch.no_grad():
                        clue_samples, clue_responses = self._generate_clue_phase_for_training(
                            sample, model, num_rounds, requires_grad=False
                        )
                
                # Minimal memory cleanup (only clear CUDA cache, don't change logic)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Phase 2: Generate God's perspective decision - use no_grad if not training decision phase
                if training_phase in ["decision", "both"]:
                    if self.accelerator.process_index == 0:
                        print(f"[TRAINING] Generating God decision phase WITH gradients")
                    decision_samples, decision_responses = self._generate_god_decision_phase_for_training(
                        sample, clue_responses, model, num_rounds, requires_grad=True
                    )
                else:
                    if self.accelerator.process_index == 0:
                        print(f"[TRAINING] Generating God decision phase WITHOUT gradients (inference only)")
                    with torch.no_grad():
                        decision_samples, decision_responses = self._generate_god_decision_phase_for_training(
                            sample, clue_responses, model, num_rounds, requires_grad=False
                        )
                
                # Collect all samples from both phases based on training_phase
                # IMPORTANT: Always include decision samples in all_training_samples for reward calculation,
                # even when only training clue phase, because clue rewards depend on decision results
                if training_phase == "clue":
                    # Include clue samples for training AND decision samples for reward calculation
                    all_training_samples.extend(clue_samples + decision_samples)
                    all_phase_labels.extend(['clue'] * len(clue_samples) + ['god_decision_for_clue_reward'] * len(decision_samples))
                    sample_count = len(clue_samples) + len(decision_samples)
                elif training_phase == "decision":
                    # Only include God decision samples for training (num_generations samples)
                    all_training_samples.extend(decision_samples)
                    all_phase_labels.extend(['god_decision'] * len(decision_samples))
                    sample_count = len(decision_samples)
                else:  # "both"
                    # Include both phases for training
                    all_training_samples.extend(clue_samples + decision_samples)
                    all_phase_labels.extend(['clue'] * len(clue_samples) + ['god_decision'] * len(decision_samples))
                    sample_count = len(clue_samples) + len(decision_samples)
                
                # Store metadata for reward calculation (always include both phases for reward calculation)
                game_metadata = {
                    'game_data': game_data,
                    'clue_responses': clue_responses,
                    'decision_responses': decision_responses,
                    'clue_samples': clue_samples,
                    'decision_samples': decision_samples,
                    'training_phase': training_phase,
                    'base_training_phase': base_training_phase  # Store original phase for logging
                }
                # Each sample gets its own metadata copy
                all_game_metadata.extend([game_metadata] * sample_count)
                
                # Memory monitoring after each game
                if hasattr(torch.cuda, 'memory_allocated') and self.accelerator.process_index == 0:
                    memory_after_game = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    print(f"[MEMORY] After processing game {i+1}: {memory_after_game:.2f} GB")
            
            # Process all samples together for GRPO training
            processed_inputs = self._generate_and_score_completions_for_clevr_two_phase(
                all_training_samples, all_phase_labels, all_game_metadata, model, training_phase
            )
            
            # Cache the results
            if not hasattr(self, '_clevr_spotdiff_cache'):
                self._clevr_spotdiff_cache = {}
            self._clevr_spotdiff_cache[buffer_key] = processed_inputs
            
            # Clean up old cache entries periodically to prevent memory accumulation
            if len(self._clevr_spotdiff_cache) > self.args.gradient_accumulation_steps * 2:
                # Keep only recent entries
                keys_to_keep = [f"clevr_spotdiff_{i}" for i in range(self.args.gradient_accumulation_steps)]
                keys_to_remove = [k for k in self._clevr_spotdiff_cache.keys() if k not in keys_to_keep]
                for key in keys_to_remove:
                    if key in self._clevr_spotdiff_cache:
                        del self._clevr_spotdiff_cache[key]
                if self.accelerator.process_index == 0:
                    print(f"[MEMORY] Cleaned up {len(keys_to_remove)} old cache entries")
                
        else:
            # Use cached results
            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] Using CACHED results (global_step={self.state.global_step}, buffer_key={buffer_key})")
            processed_inputs = getattr(self, '_clevr_spotdiff_cache', {}).get(buffer_key)
            if processed_inputs is None:
                # Fallback: force regeneration by setting global_step condition to True
                if self.accelerator.process_index == 0:
                    print("[WARNING] Cache missing, forcing regeneration")
                # Regenerate without recursion by processing current inputs directly
                all_training_samples = []
                all_phase_labels = []
                all_game_metadata = []
                
                # Process only the first game to avoid OOM
                if len(inputs) > 1:
                    inputs = [inputs[0]]
                
                for i, sample in enumerate(inputs):
                    game_data = sample["game_data"]
                    num_players = game_data["num_players"]
                    num_rounds = getattr(self, '_clevr_num_rounds', 2)
                    
                    # Phase 1: Generate clues
                    clue_samples, clue_responses = self._generate_clue_phase_for_training(sample, model, num_rounds)
                    
                    # Phase 2: Generate decisions
                    decision_samples, decision_responses = self._generate_decision_phase_for_training(
                        sample, clue_responses, model, num_rounds
                    )
                    
                    # Collect samples
                    all_training_samples.extend(clue_samples + decision_samples)
                    all_phase_labels.extend(['clue'] * len(clue_samples) + ['decision'] * len(decision_samples))
                    
                    game_metadata = {
                        'game_data': game_data,
                        'clue_responses': clue_responses,
                        'decision_responses': decision_responses,
                        'clue_samples': clue_samples,
                        'decision_samples': decision_samples
                    }
                    all_game_metadata.extend([game_metadata] * (len(clue_samples) + len(decision_samples)))
                
                # Process samples for GRPO training
                processed_inputs = self._generate_and_score_completions_for_clevr_two_phase(
                    all_training_samples, all_phase_labels, all_game_metadata, model, training_phase
                )
                
                # Cache the results
                if not hasattr(self, '_clevr_spotdiff_cache'):
                    self._clevr_spotdiff_cache = {}
                self._clevr_spotdiff_cache[buffer_key] = processed_inputs
        
        # Final memory monitoring
        if hasattr(torch.cuda, 'memory_allocated') and self.accelerator.process_index == 0:
            memory_final = torch.cuda.memory_allocated(device) / 1024**3  # GB
            print(f"[MEMORY] After CLEVR processing: {memory_final:.2f} GB")
        
        return processed_inputs
    
    def _generate_clue_phase_for_training(self, sample: dict, model, num_rounds: int, requires_grad=False) -> tuple[list, list]:
        """Generate clue phase samples that will participate in training with sequential clue passing"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator(num_rounds=num_rounds)
        
        game_data = sample["game_data"]
        num_players = game_data["num_players"]
        
        clue_samples = []
        clue_responses = []
        all_player_clues = []
        all_player_thinking = {i: [] for i in range(1, num_players + 1)}
        
        # Use the requires_grad parameter passed from caller (respects Interactive mode)
        train_clue_phase = requires_grad
        use_no_grad_for_clue = not requires_grad
        
        if self.accelerator.process_index == 0:
            print(f"[CLUE PHASE] Training: {train_clue_phase}, Using no_grad: {use_no_grad_for_clue}, requires_grad: {requires_grad}")
        
        # Generate clues sequentially so each player can see previous real clues
        for round_num in range(1, num_rounds + 1):
            for player_id in range(1, num_players + 1):
                # Create integrated previous clues text with REAL clues from previous players
                # AND this player's own thinking history from previous rounds
                previous_clues_text = self._build_integrated_clues_text_with_own_thinking(
                    all_player_clues, all_player_thinking[player_id], player_id
                )
                
                # Create clue sample for training
                clue_sample = self._create_clue_training_sample(sample, player_id, round_num, previous_clues_text, num_rounds)
                clue_samples.append(clue_sample)
                
                # Generate REAL clue response for this player right now
                real_clue_response = self._generate_single_clue_for_training(clue_sample, model, requires_grad)
                clue_responses.append(real_clue_response)
                
                if self.accelerator.process_index == 0:
                    print(f"Round {round_num} - Player {player_id} Generated Real Clue:")
                    print(f"  {real_clue_response}")
                
                # Extract thinking and clue from the real response
                thinking = self._clevr_generator.extract_thinking_from_clue(real_clue_response)
                if thinking:
                    all_player_thinking[player_id].append(f"Round {round_num}: {thinking}")
                
                # Extract clue text for next players to see
                clue_text = self._extract_clue_text_from_response(real_clue_response)
                clue_entry = f"Round {round_num} - Player {player_id}: {clue_text}"
                all_player_clues.append(clue_entry)
                
                # Memory cleanup for non-training phase
                if use_no_grad_for_clue and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Store the thinking history for use in decision phase
        self._current_game_thinking = all_player_thinking
        
        return clue_samples, clue_responses
    
    def _generate_single_clue_for_training(self, clue_sample: dict, model, requires_grad=True) -> str:
        """Generate a single clue response using the model"""
        device = self.accelerator.device
        
        # Debug: Print clue input information
        if self.accelerator.process_index == 0:
            player_id = clue_sample.get("player_id", "unknown")
            round_num = clue_sample.get("round_num", "unknown")
            image_paths = clue_sample.get("image_path", [])
            print(f"\n[CLUE INPUT]  Player {player_id} Round {round_num}")
            print("Available keys in clue_sample:", list(clue_sample.keys()))
            if 'problem' in clue_sample:
                print("Raw prompt text  ↓↓↓")
                print(clue_sample['problem'])
            else:
                print("No 'problem' key found!")
                if 'prompt' in clue_sample:
                    print("Found 'prompt' key instead:")
                    prompt_content = clue_sample['prompt']
                    if isinstance(prompt_content, list) and len(prompt_content) > 0:
                        user_content = prompt_content[0].get('content', [])
                        for item in user_content:
                            if item.get('type') == 'text':
                                print("Text content:", item.get('text', ''))
                                break
            print(f"Image path: {image_paths}")
            print()
        
        # Prepare inputs similar to _generate_single_completion but for clue generation
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [clue_sample])
        
        # Handle image loading
        if "image_path" in clue_sample and clue_sample["image_path"] is not None:
            images = [PIL.Image.open(p) for p in clue_sample["image_path"]]
        else:
            images = []

        for img in images:
            try:
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
        
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device and handle dtypes
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'input_ids' in prompt_inputs:
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(dtype=torch.long)
            if 'attention_mask' in prompt_inputs:
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(dtype=torch.long)
        
        # Generate clue with appropriate generation config
        if requires_grad:
            # Normal training mode - gradients will be computed
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                clue_generation_config = GenerationConfig(
                    max_new_tokens=1024,  # Sufficient space for detailed thinking
                    do_sample=True,
                    temperature=self.generation_config.temperature,
                    min_new_tokens=20,   # Ensure sufficient thinking before stopping
                    pad_token_id=self.generation_config.pad_token_id,
                    eos_token_id=self.generation_config.eos_token_id,
                )
                if hasattr(self.vlm_module, "get_eos_token_id"):
                    clue_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                
                generated_ids = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=clue_generation_config,
                    logits_processor=self._rollout_logits_processor(),
                )

                # Extract only the generated part
                prompt_length = prompt_inputs["input_ids"].size(1)
                if not self.vlm_module.is_embeds_input():
                    clue_ids = generated_ids[:, prompt_length:]
                else:
                    clue_ids = generated_ids

                # Decode the clue
                clue_text = self.processing_class.batch_decode(clue_ids, skip_special_tokens=True)[0]

                return clue_text
        else:
            # Inference mode - no gradients needed
            with torch.no_grad():
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    clue_generation_config = GenerationConfig(
                        max_new_tokens=1024,  # Sufficient space for detailed thinking
                        do_sample=True,
                        temperature=self.generation_config.temperature,
                        min_new_tokens=20,   # Ensure sufficient thinking before stopping
                        pad_token_id=self.generation_config.pad_token_id,
                        eos_token_id=self.generation_config.eos_token_id,
                    )
                    if hasattr(self.vlm_module, "get_eos_token_id"):
                        clue_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                    
                    generated_ids = unwrapped_model.generate(
                        **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                        generation_config=clue_generation_config,
                        logits_processor=self._rollout_logits_processor(),
                    )

                    # Extract only the generated part
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    if not self.vlm_module.is_embeds_input():
                        clue_ids = generated_ids[:, prompt_length:]
                    else:
                        clue_ids = generated_ids

                    # Decode the clue
                    clue_text = self.processing_class.batch_decode(clue_ids, skip_special_tokens=True)[0]

                    return clue_text
    
    def _extract_clue_text_from_response(self, response: str) -> str:
        """Extract clue text from model response with flexible format support"""
        import re
        
        # Try boxed format first
        boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Fallback: try old answer tags format for backward compatibility
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: try to find <answer> and extract everything after it (like clue phase)
        answer_start_match = re.search(r'<answer>\s*(.*)', response, re.DOTALL)
        if answer_start_match:
            answer_content = answer_start_match.group(1).strip()
            # Clean up any potential closing tags or extra content
            answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
            if answer_content:
                return answer_content
        
        # If no answer format found, return placeholder instead of whole response
        return "No valid clue provided."
    
    def _create_clue_training_sample(self, sample: dict, player_id: int, round_num: int, 
                                    previous_clues_text: str, num_rounds: int) -> dict:
        """Create a clue sample formatted for GRPO training"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator(num_rounds=num_rounds)
        
        game_data = sample["game_data"]
        
        # Create clue phase sample
        clue_sample_data = self._clevr_generator.format_clue_phase_sample(
            game_data, player_id, round_num, previous_clues_text, ""
        )
        
        # Get player's image path
        try:
            player_image_path = game_data["player_images"][player_id - 1]
        except (IndexError, KeyError):
            player_image_path = "/dev/null"
        
        # Format for GRPO training
        training_sample = {
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "text": None},
                    {"type": "text", "text": clue_sample_data["conversations"][0]["value"].replace("<image>\n", "")}
                ]
            }],
            "image_path": [player_image_path],
            "problem": clue_sample_data["conversations"][0]["value"].replace("<image>\n", ""),
            "solution": clue_sample_data["conversations"][1]["value"],
            "accu_reward_method": "clevr_spotdiff_clue",
            "game_data": game_data,
            "player_id": player_id,
            "round_num": round_num,
            "phase": "clue",
            "metadata": sample.get("metadata", {})
        }
        
        return training_sample
    
    def _generate_decision_phase_for_training(self, sample: dict, clue_responses: list, 
                                            model, num_rounds: int, requires_grad=False) -> tuple[list, list]:
        """Generate decision phase samples that will participate in training"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator(num_rounds=num_rounds)
        
        game_data = sample["game_data"]
        num_players = game_data["num_players"]
        
        # Check if decision phase should be trained
        train_decision_phase = (self.clevr_training_phase in ['decision', 'both'])
        use_no_grad_for_decision = not train_decision_phase
        
        if self.accelerator.process_index == 0:
            print(f"[DECISION PHASE] Training: {train_decision_phase}, Using no_grad: {use_no_grad_for_decision}")
        
        # Build integrated clues text from REAL clue responses
        all_clues = self._build_all_clues_from_responses(clue_responses, num_players, num_rounds)
        
        decision_samples = []
        decision_responses = []
        
        # Create decision phase samples for each player
        for player_id in range(1, num_players + 1):
            # Build decision clues text that includes this player's thinking history
            player_thinking = getattr(self, '_current_game_thinking', {}).get(player_id, [])
            integrated_decision_clues = self._build_decision_clues_with_thinking(all_clues, player_thinking, player_id)
            
            decision_sample = self._create_decision_training_sample(sample, player_id, integrated_decision_clues)
            decision_samples.append(decision_sample)
            
            # Generate REAL decision response
            real_decision_response = self._generate_single_completion(decision_sample, model, requires_grad)[0]  # Only need completion
            decision_responses.append(real_decision_response)
            
            if self.accelerator.process_index == 0:
                print(f"Player {player_id} Generated Real Decision:")
                print(f"  {real_decision_response}")
            
            # Memory cleanup for non-training phase
            if use_no_grad_for_decision and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        return decision_samples, decision_responses
    
    def _generate_god_decision_phase_for_training(self, sample: dict, clue_responses: list, 
                                                model, num_rounds: int, requires_grad=False) -> tuple[list, list]:
        """Generate God's perspective decision phase samples for training (batch generation like normal GRPO)"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator(num_rounds=num_rounds)
        
        game_data = sample["game_data"]
        num_players = game_data["num_players"]
        
        # Use the requires_grad parameter passed from caller (respects Interactive mode)
        train_decision_phase = requires_grad
        
        if self.accelerator.process_index == 0:
            print(f"[GOD DECISION PHASE] Training: {train_decision_phase}, requires_grad: {requires_grad}")
            print(f"[GOD DECISION] Batch generating {self.num_generations} answers for God's perspective")
        
        # Build integrated clues text from REAL clue responses (no thinking for God view)
        all_clues = self._build_all_clues_from_responses(clue_responses, num_players, num_rounds)
        
        # Create ONE God's perspective decision sample
        god_decision_sample = self._create_god_decision_training_sample(sample, all_clues)
        
        # Create batch with num_generations copies of the same God sample (like normal GRPO)
        batch_samples = []
        for gen_idx in range(self.num_generations):
            sample_copy = god_decision_sample.copy()
            sample_copy['generation_idx'] = gen_idx
            batch_samples.append(sample_copy)
        
        # Batch generate like normal GRPO
        device = self.accelerator.device
        
        # Prepare inputs for batch generation
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, batch_samples)
        
        # Handle image loading (all samples use same God perspective image)
        original_image_path = game_data["comparison_data"]["original_image_path"]
        images = [PIL.Image.open(original_image_path) for _ in range(self.num_generations)]
        
        for img in images:
            try:
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
        
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device and handle dtypes
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'input_ids' in prompt_inputs:
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(dtype=torch.long)
            if 'attention_mask' in prompt_inputs:
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(dtype=torch.long)
        
        # Batch generate with appropriate generation config based on requires_grad
        if requires_grad:
            # Normal training mode - gradients will be computed
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                decision_generation_config = GenerationConfig(
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=self.generation_config.temperature,
                    min_new_tokens=100,
                    repetition_penalty=1.1,
                    pad_token_id=self.generation_config.pad_token_id,
                    eos_token_id=self.generation_config.eos_token_id,
                    early_stopping=False,
                )
                # Commented out to keep eos_token_id as None to prevent early stopping
                # if hasattr(self.vlm_module, "get_eos_token_id"):
                #     decision_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                
                generated_ids = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=decision_generation_config,
                    logits_processor=self._rollout_logits_processor(),
                )

                prompt_length = prompt_inputs["input_ids"].size(1)
                if not self.vlm_module.is_embeds_input():
                    completion_ids = generated_ids[:, prompt_length:]
                else:
                    completion_ids = generated_ids
        else:
            # Inference mode - no gradients needed
            with torch.no_grad():
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    decision_generation_config = GenerationConfig(
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.generation_config.temperature,
                        min_new_tokens=100,
                        repetition_penalty=1.1,
                        pad_token_id=self.generation_config.pad_token_id,
                        eos_token_id=self.generation_config.eos_token_id,
                        early_stopping=False,
                    )
                    # Commented out to keep consistent with requires_grad=True branch
                    # if hasattr(self.vlm_module, "get_eos_token_id"):
                    #     decision_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                    
                    generated_ids = unwrapped_model.generate(
                        **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                        generation_config=decision_generation_config,
                        logits_processor=self._rollout_logits_processor(),
                    )

                    prompt_length = prompt_inputs["input_ids"].size(1)
                    if not self.vlm_module.is_embeds_input():
                        completion_ids = generated_ids[:, prompt_length:]
                    else:
                        completion_ids = generated_ids
        
        # Decode all completions at once
        decision_responses = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        if self.accelerator.process_index == 0:
            print(f"[GOD DECISION] Batch generated {len(decision_responses)} answers")
            
            # Print the first God decision input prompt for debugging
            if batch_samples:
                first_sample = batch_samples[0]
                if "prompt" in first_sample and first_sample["prompt"]:
                    user_prompt = first_sample["prompt"][0]["content"]
                    if len(user_prompt) >= 2 and user_prompt[1]["type"] == "text":
                        print(f"[DEBUG] GOD DECISION INPUT PROMPT:")
                        print(f"[DEBUG] {'='*60}")
                        print(user_prompt[1]["text"])
                        print(f"[DEBUG] {'='*60}")
            
            for i, response in enumerate(decision_responses):
                print(f"God Decision Generation {i + 1}: {response[:100]}...")
        
        return batch_samples, decision_responses
    
    def _create_god_decision_training_sample(self, sample: dict, all_clues: str) -> dict:
        """Create a God's perspective decision sample formatted for GRPO training"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        game_data = sample["game_data"]
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator()
        
        # Create God's perspective decision phase sample
        god_decision_sample_data = self._clevr_generator.format_god_decision_phase_sample(
            game_data, all_clues
        )
        
        # Get original image path for God's perspective
        original_image_path = game_data["comparison_data"]["original_image_path"]
        
        # Format for GRPO training
        training_sample = {
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "text": None},
                    {"type": "text", "text": god_decision_sample_data["conversations"][0]["value"].replace("<image>\n", "")}
                ]
            }],
            "image_path": [original_image_path],  # God sees original image
            "problem": god_decision_sample_data["conversations"][0]["value"].replace("<image>\n", ""),
            "solution": god_decision_sample_data["conversations"][1]["value"],
            "accu_reward_method": "clevr_spotdiff_god_decision",  # New reward method for God perspective
            "game_data": game_data,
            "correct_spy": god_decision_sample_data["correct_spy"],
            "phase": "god_decision",
            "metadata": sample.get("metadata", {})
        }
        
        return training_sample
    
    def _build_all_clues_from_responses(self, clue_responses: list[str], num_players: int, num_rounds: int) -> str:
        """Build integrated clues text from real clue responses"""
        all_clues_lines = []
        
        response_idx = 0
        for round_num in range(1, num_rounds + 1):
            all_clues_lines.append(f"Round {round_num}:")
            for player_id in range(1, num_players + 1):
                if response_idx < len(clue_responses):
                    clue_text = self._extract_clue_text_from_response(clue_responses[response_idx])
                    all_clues_lines.append(f"Player {player_id}: {clue_text}")
                    response_idx += 1
                else:
                    all_clues_lines.append(f"Player {player_id}: No clue provided")
            all_clues_lines.append("")  # Empty line between rounds
        
        return "\n".join(all_clues_lines).strip()
    
    def _build_decision_clues_with_thinking(self, all_clues: str, player_thinking: list, player_id: int) -> str:
        """Build decision clues text that includes the current player's thinking history
        CRITICAL: Other players can only see ANSWER parts, not thinking!"""
        if not all_clues and not player_thinking:
            return ""
        
        # Parse the existing all_clues text
        clues_lines = all_clues.split('\n') if all_clues else []
        
        # Parse thinking history by round
        thinking_by_round = {}
        for thinking_entry in player_thinking:
            if thinking_entry.startswith("Round "):
                round_num_str = thinking_entry.split(":")[0].strip()
                thinking_text = thinking_entry.split(":", 1)[1].strip()
                thinking_by_round[round_num_str] = thinking_text
        
        # Build integrated decision text with thinking appearing before current player's clue
        # CRITICAL FIX: Clean other players' content to hide their thinking
        integrated_lines = []
        current_round = None
        
        for line in clues_lines:
            if line.strip().startswith("Round ") and ":" in line:
                # This is a round header
                current_round = line.strip().rstrip(":")
                integrated_lines.append(line)
            elif line.strip().startswith(f"Player {player_id}:"):
                # This is current player's clue line - add thinking before it
                if current_round and current_round in thinking_by_round:
                    integrated_lines.append(f"Player {player_id} (my thinking): {thinking_by_round[current_round]}")
                integrated_lines.append(line)
            elif line.strip().startswith("Player ") and ":" in line:
                # This is another player's clue line - CRITICAL: clean it to hide thinking
                cleaned_line = self._clean_other_player_clue_line(line)
                integrated_lines.append(cleaned_line)
            else:
                # This is empty line or other content - only add if it doesn't contain thinking
                if not self._contains_thinking_content(line):
                    integrated_lines.append(line)
        
        return "\n".join(integrated_lines)
    
    def _clean_other_player_clue_line(self, line: str) -> str:
        """Clean other player's clue line to show only answer content (hide thinking)"""
        import re
        
        # Check if this line is a player clue line
        if not line.strip().startswith("Player ") or ":" not in line:
            return line
        
        # Extract player info and content
        parts = line.split(":", 1)
        if len(parts) != 2:
            return line
            
        player_info = parts[0].strip()
        content = parts[1].strip()
        
        # Extract only answer content from the content
        clean_content = self._extract_answer_only_from_text(content)
        
        return f"{player_info}: {clean_content}"
    
    def _contains_thinking_content(self, line: str) -> bool:
        """Check if a line contains thinking content that should be hidden"""
        line_lower = line.strip().lower()
        
        # Check for thinking-related patterns
        thinking_patterns = [
            "thinking):",
            "<think>",
            "</think>",
            "(my thinking)",
            "my thinking:"
        ]
        
        return any(pattern in line_lower for pattern in thinking_patterns)
    
    def _create_decision_training_sample(self, sample: dict, player_id: int, all_clues: str) -> dict:
        """Create a decision sample formatted for GRPO training"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        game_data = sample["game_data"]
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator()
        
        # Create decision phase sample
        decision_sample_data = self._clevr_generator.format_decision_phase_sample(
            game_data, player_id, all_clues, ""
        )
        
        # Get player's image path
        try:
            player_image_path = game_data["player_images"][player_id - 1]
        except (IndexError, KeyError):
            player_image_path = "/dev/null"
        
        # Format for GRPO training
        training_sample = {
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "text": None},
                    {"type": "text", "text": decision_sample_data["conversations"][0]["value"].replace("<image>\n", "")}
                ]
            }],
            "image_path": [player_image_path],
            "problem": decision_sample_data["conversations"][0]["value"].replace("<image>\n", ""),
            "solution": decision_sample_data["conversations"][1]["value"],
            "accu_reward_method": "clevr_spotdiff_decision",
            "game_data": game_data,
            "player_id": player_id,
            "correct_spy": decision_sample_data["correct_spy"],
            "phase": "decision",
            "metadata": sample.get("metadata", {})
        }
        
        return training_sample
    
    def _generate_and_score_completions_for_clevr_two_phase(self, training_samples: list[dict], 
                                                          phase_labels: list[str], 
                                                          game_metadata: list[dict], model, training_phase) -> dict:
        """Generate and score completions for both clue and decision phases"""
        device = self.accelerator.device
        
        # Get pre-generated completions from metadata (clue completions were already generated)
        # We need to create multimodal inputs for all samples for loss computation
        all_completions = []
        all_multimodal_caches = []
        all_precomputed_logps = []  # Store precomputed logps to avoid second forward pass
        
        for i, (sample, phase) in enumerate(zip(training_samples, phase_labels)):
            game_meta = game_metadata[i]
            
            if phase == "clue":
                # Use pre-generated clue completion from game_metadata
                clue_idx = len([j for j in range(i) if phase_labels[j] == "clue"])  # Get clue index
                if clue_idx < len(game_meta["clue_responses"]):
                    completion = game_meta["clue_responses"][clue_idx]
                    # Compute logps immediately with the current model state
                    logps_data = self._compute_and_cache_logps_for_sample(sample, completion, model, device)
                    all_precomputed_logps.append(logps_data)
                else:
                    # Fallback: generate if missing
                    completion, multimodal_cache, logps_data = self._generate_single_completion_with_logps(sample, model)
                    all_multimodal_caches.append(multimodal_cache)
                    all_completions.append(completion)
                    all_precomputed_logps.append(logps_data)
                    continue
                    
            else:  # god_decision phase (including god_decision_for_clue_reward)
        # Use pre-generated God decision completion from game_metadata
                god_decision_idx = len([j for j in range(i) if phase_labels[j] in ["god_decision", "god_decision_for_clue_reward"]])  # Get god decision index
                if god_decision_idx < len(game_meta["decision_responses"]):
                    completion = game_meta["decision_responses"][god_decision_idx]
                    # Compute logps immediately with the current model state
                    logps_data = self._compute_and_cache_logps_for_sample(sample, completion, model, device)
                    all_precomputed_logps.append(logps_data)
                else:
                    # Fallback: generate if missing
                    completion, multimodal_cache, logps_data = self._generate_single_completion_with_logps(sample, model)
                    all_multimodal_caches.append(multimodal_cache)
                    all_completions.append(completion)
                    all_precomputed_logps.append(logps_data)
                    continue
            
            # For pre-generated completions, we still need to create multimodal inputs for loss computation
            # Process the sample to get multimodal inputs without regenerating the text
            multimodal_cache = self._create_multimodal_cache_for_sample(sample, device)
            all_multimodal_caches.append(multimodal_cache)
            all_completions.append(completion)
        
        # Calculate rewards for both phases after all generations are complete
        all_rewards = self._calculate_two_phase_rewards(
            training_samples, all_completions, phase_labels, game_metadata
        )
        
        # Condensed reward analysis - key info printed on rank 0 only
        if self.accelerator.process_index == 0:
            print(f"[REWARD] GPU-0 Summary for {len(training_samples)} samples:")
            
            # Clue-stage rewards (rank 0)
            clue_rewards = [all_rewards[i] for i, p in enumerate(phase_labels) if p == "clue"]
            if clue_rewards:
                print(f"[CLUE] Rewards: {[f'{r:.3f}' for r in clue_rewards]}")
            
            # Decision stage (God decision on rank 0)
            decision_indices = [i for i, p in enumerate(phase_labels) if p in ["god_decision", "god_decision_for_clue_reward"]]
            if decision_indices:
                # Show every decision's vote
                god_votes = []
                for idx in decision_indices:
                    completion = all_completions[idx]
                    if hasattr(self, '_clevr_generator'):
                        vote_info = self._clevr_generator.extract_vote_from_decision(completion)
                        voted_spy = vote_info.get('voted_spy', 'None') if vote_info else 'None'
                        god_votes.append(voted_spy)
                
                decision_rewards = [all_rewards[i] for i in decision_indices]
                print(f"[GOD DECISION] Votes: {god_votes}")
                print(f"[GOD DECISION] Raw rewards: {[f'{r:.3f}' for r in decision_rewards]}")
            
            print("="*50)
        
        # Condensed game-result output - rank 0 only
        if len(training_samples) > 0 and self.accelerator.process_index == 0:
            sample_game_data = training_samples[0].get('game_data', {})
            game_id = sample_game_data.get('game_id', 'unknown')
            spy_player = sample_game_data.get('spy_player', 'unknown')
            print(f"[GAME] {game_id}, Spy: Player {spy_player}")
        
        # Format results for GRPO training using precomputed logps (avoid second forward pass)
        # **************************************************
        # ABLATION STUDY: Add phase_labels for clue loss zeroing
        # **************************************************
        processed_results = self._format_two_phase_results_with_precomputed_logps(
            training_samples, all_completions, all_multimodal_caches, all_rewards, all_precomputed_logps, device
        )
        # Add phase information for clue loss zeroing
        processed_results["phase_labels"] = phase_labels
        # **************************************************
        # END ABLATION STUDY MODIFICATION
        # **************************************************
        return processed_results
    
    def _calculate_two_phase_rewards(self, training_samples: list[dict], completions: list[str], 
                                   phase_labels: list[str], game_metadata: list[dict]) -> list[float]:
        """Calculate rewards for both clue and God decision phases"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator, _calculate_enhanced_two_phase_rewards_with_god_decision
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator()
        
        # Group samples by game
        games = {}
        for i, (sample, completion, phase, metadata) in enumerate(zip(training_samples, completions, phase_labels, game_metadata)):
            game_id = sample["game_data"]["game_id"]
            if game_id not in games:
                games[game_id] = {
                    "clue_samples": [],
                    "god_decision_samples": [],
                    "clue_completions": [],
                    "god_decision_completions": [],
                    "clue_indices": [],
                    "god_decision_indices": [],
                    "game_data": sample["game_data"]
                }
            
            if phase == "clue":
                games[game_id]["clue_samples"].append(sample)
                games[game_id]["clue_completions"].append(completion)
                games[game_id]["clue_indices"].append(i)
            elif phase in ["god_decision", "god_decision_for_clue_reward"]:  # Handle both god_decision types
                games[game_id]["god_decision_samples"].append(sample)
                games[game_id]["god_decision_completions"].append(completion)
                games[game_id]["god_decision_indices"].append(i)
            else:
                # Handle any unexpected phase types
                if self.accelerator.process_index == 0:
                    print(f"[WARNING] Unknown phase type: {phase}, treating as god_decision")
                games[game_id]["god_decision_samples"].append(sample)
                games[game_id]["god_decision_completions"].append(completion)
                games[game_id]["god_decision_indices"].append(i)
        
        # Initialize rewards list
        all_rewards = [0.0] * len(training_samples)
        
        # Store original rewards for logging before normalization
        all_original_clue_rewards = []
        all_original_decision_rewards = []
        
        # Collect all clue metrics for aggregation
        all_clue_metrics = []
        
        # Calculate rewards for each game using enhanced method with God decision
        for game_id, game_info in games.items():
            game_data = game_info["game_data"]
            clue_completions = game_info["clue_completions"]
            god_decision_completions = game_info["god_decision_completions"]
            
            # Use enhanced reward calculation with God's perspective decision
            clue_rewards, god_decision_rewards, clue_metrics = _calculate_enhanced_two_phase_rewards_with_god_decision(
                self._clevr_generator, game_data, clue_completions, god_decision_completions
            )
            
            # Store clue metrics for aggregation
            all_clue_metrics.append(clue_metrics)
            
            # Store original rewards for logging (before normalization)
            all_original_clue_rewards.extend(clue_rewards)
            all_original_decision_rewards.extend(god_decision_rewards)
            
            # Skip group normalization for clue rewards - strategic clue rewards are already zero-sum by design
            clue_rewards_normalized = clue_rewards
            if self.accelerator.process_index == 0:
                print(f"[STRATEGIC CLUE] Skipping normalization for clue phase (already zero-sum by design)")
                print(f"[STRATEGIC CLUE] Raw clue rewards: {[f'{r:.3f}' for r in clue_rewards]}")
            
            # For God decision rewards, apply group normalization across generations
            if len(god_decision_rewards) > 1:
                # God decision rewards are from multiple generations of the same decision task
                god_decision_rewards_normalized = self._apply_group_normalization(god_decision_rewards, "god_decision", game_id)
                if self.accelerator.process_index == 0:
                    print(f"[GOD DECISION] Applied group normalization across {len(god_decision_rewards)} generations")
                    print(f"[GOD DECISION] Raw rewards: {[f'{r:.3f}' for r in god_decision_rewards]}")
                    print(f"[GOD DECISION] Normalized rewards: {[f'{r:.3f}' for r in god_decision_rewards_normalized]}")
            else:
                god_decision_rewards_normalized = god_decision_rewards  # Single generation, no normalization needed
            
            # Assign normalized rewards back to correct positions (these are used for training)
            for reward, idx in zip(clue_rewards_normalized, game_info["clue_indices"]):
                all_rewards[idx] = reward
            for reward, idx in zip(god_decision_rewards_normalized, game_info["god_decision_indices"]):
                all_rewards[idx] = reward
            
            # Debug output showing both original and normalized rewards
            if self.accelerator.process_index == 0:
                print(f"Game {game_id} rewards:")
                print(f"  Clue phase (original): {[f'{r:.3f}' for r in clue_rewards]}")
                print(f"  Clue phase (normalized): {[f'{r:.3f}' for r in clue_rewards_normalized]}")
                print(f"  God decision phase (original): {[f'{r:.3f}' for r in god_decision_rewards]}")
                print(f"  God decision phase (normalized): {[f'{r:.3f}' for r in god_decision_rewards_normalized]}")
        
        # Aggregate clue metrics across all games for wandb logging
        if all_clue_metrics:
            # Calculate mean values across all games
            aggregated_metrics = {}
            metric_keys = all_clue_metrics[0].keys()
            
            for key in metric_keys:
                values = [metrics[key] for metrics in all_clue_metrics if key in metrics]
                if values:
                    # Only aggregate numerical values, skip dict/list types
                    numerical_values = []
                    for val in values:
                        if isinstance(val, (int, float)):
                            numerical_values.append(val)
                        elif isinstance(val, dict) and len(val) > 0:
                            # For dict values, try to get meaningful numerical summary
                            if key == 'god_vote_counts':
                                # Sum of all vote counts
                                numerical_values.append(sum(val.values()))
                            elif key == 'god_suspicion_scores':
                                # Average suspicion score
                                if val:
                                    numerical_values.append(sum(val.values()) / len(val.values()))
                    
                    # Only log if we have numerical values
                    if numerical_values:
                        aggregated_metrics[f"clue_{key}"] = sum(numerical_values) / len(numerical_values)
            
            # Log aggregated clue metrics
            for metric_name, metric_value in aggregated_metrics.items():
                self._metrics[metric_name].append(metric_value)
            
            if self.accelerator.process_index == 0:
                print(f"[CLUE METRICS] Logged {len(aggregated_metrics)} aggregated metrics to wandb:")
                for metric_name, metric_value in aggregated_metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        # Log original rewards to metrics (for wandb visualization)
        original_clue_mean = 0.0
        original_clue_std = 0.0
        original_decision_mean = 0.0
        original_decision_std = 0.0
        
        if all_original_clue_rewards:
            original_clue_mean = sum(all_original_clue_rewards) / len(all_original_clue_rewards)
            original_clue_std = (sum([(r - original_clue_mean)**2 for r in all_original_clue_rewards]) / len(all_original_clue_rewards))**0.5
            self._metrics["reward_original_clue_mean"].append(original_clue_mean)
            self._metrics["reward_original_clue_std"].append(original_clue_std)
            self._metrics["reward_original_clue_min"].append(min(all_original_clue_rewards))
            self._metrics["reward_original_clue_max"].append(max(all_original_clue_rewards))
            
        if all_original_decision_rewards:
            original_decision_mean = sum(all_original_decision_rewards) / len(all_original_decision_rewards)
            original_decision_std = (sum([(r - original_decision_mean)**2 for r in all_original_decision_rewards]) / len(all_original_decision_rewards))**0.5
            self._metrics["reward_original_decision_mean"].append(original_decision_mean)
            self._metrics["reward_original_decision_std"].append(original_decision_std)
            self._metrics["reward_original_decision_min"].append(min(all_original_decision_rewards))
            self._metrics["reward_original_decision_max"].append(max(all_original_decision_rewards))
        
        # Log overall original reward statistics  
        all_original_rewards = all_original_clue_rewards + all_original_decision_rewards
        if all_original_rewards:
            original_overall_mean = sum(all_original_rewards) / len(all_original_rewards)
            original_overall_std = (sum([(r - original_overall_mean)**2 for r in all_original_rewards]) / len(all_original_rewards))**0.5
            self._metrics["reward_original_overall_mean"].append(original_overall_mean)
            self._metrics["reward_original_overall_std"].append(original_overall_std)
            
            # Also log to the standard "reward" metric that wandb tracks
            self._metrics["reward"].append(original_overall_mean)
            self._metrics["reward_std"].append(original_overall_std)
            
            if self.accelerator.process_index == 0:
                print(f"Original reward stats - Mean: {original_overall_mean:.3f}, Std: {original_overall_std:.3f}")
                if all_original_clue_rewards:
                    print(f"Clue rewards - Mean: {original_clue_mean:.3f}, Std: {original_clue_std:.3f}")
                if all_original_decision_rewards:
                    print(f"Decision rewards - Mean: {original_decision_mean:.3f}, Std: {original_decision_std:.3f}")
        
        return all_rewards
    
    def _apply_group_normalization(self, rewards: list[float], phase: str, game_id: str, eps: float = 1e-8) -> list[float]:
        """
        Apply group normalization to rewards within a game phase
        
        Args:
            rewards: List of rewards for players in this phase
            phase: Phase name ("clue" or "decision") for debugging
            game_id: Game ID for debugging
            eps: Small constant to prevent division by zero
            
        Returns:
            List of normalized rewards
        """
        import torch
        
        if len(rewards) <= 1:
            return rewards
        
        # Convert to tensor for easier computation
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Calculate group statistics
        group_mean = rewards_tensor.mean()
        group_std = rewards_tensor.std(unbiased=False)  # Use population std, not sample std
        
        # Check if all values are essentially the same (more robust check)
        max_diff = (rewards_tensor.max() - rewards_tensor.min()).item()
        
        # Apply normalization with epsilon for numerical stability
        if group_std > eps and max_diff > eps:
            normalized_rewards = (rewards_tensor - group_mean) / (group_std + eps)
        else:
            # If std is too small or all values are the same, just center the rewards
            normalized_rewards = rewards_tensor - group_mean
        
        # Convert back to list
        normalized_rewards_list = normalized_rewards.tolist()
        
        # Condensed normalization output - result only
        if self.accelerator.process_index == 0 and abs(group_std) < eps:
            print(f"[NORM] {phase}: all rewards identical → normalized to 0")
        elif self.accelerator.process_index == 0:
            print(f"[NORM] {phase}: {[f'{r:.3f}' for r in rewards]} → {[f'{r:.3f}' for r in normalized_rewards_list]}")
        
        return normalized_rewards_list
    
    
    def _create_multimodal_cache_for_sample(self, sample: dict, device) -> dict:
        """Create multimodal cache for a sample without generating text completion"""
        # Prepare inputs for multimodal processing
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [sample])
        
        # Handle image loading
        if "image_path" in sample and sample["image_path"] is not None:
            images = [PIL.Image.open(p) for p in sample["image_path"]]
        else:
            images = []

        for img in images:
            try:
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
        
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device and handle dtypes
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        if hasattr(self.model, 'dtype') and self.model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
        
        # Create multimodal cache without text generation
        multimodal_cache = {}
        for key in ["pixel_values", "image_flags", "image_grid_thw"]:
            if key in prompt_inputs:
                tensor = prompt_inputs[key].squeeze(0)  # Remove batch dim
                multimodal_cache[key] = tensor
        
        return multimodal_cache
    
    def _compute_and_cache_logps_for_sample(self, sample: dict, completion: str, model, device) -> dict:
        """Compute and cache logps for a sample with pre-generated completion (avoid second forward pass)"""
        # Prepare inputs 
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [sample])
        
        # Handle image loading
        if "image_path" in sample and sample["image_path"] is not None:
            images = [PIL.Image.open(p) for p in sample["image_path"]]
        else:
            images = []

        for img in images:
            try:
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
        
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device and handle dtypes
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'input_ids' in prompt_inputs:
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(dtype=torch.long)
            if 'attention_mask' in prompt_inputs:
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(dtype=torch.long)
        
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        prompt_mask = prompt_inputs["attention_mask"].squeeze(0)
        
        # Tokenize the completion
        if hasattr(self.processing_class, 'tokenizer'):
            tokenizer = self.processing_class.tokenizer
        else:
            tokenizer = self.processing_class
            
        completion_tokens = tokenizer(
            completion,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False
        )
        completion_ids = completion_tokens["input_ids"].squeeze(0).to(device)
        completion_mask = torch.ones_like(completion_ids, dtype=torch.long, device=device)
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)  # Add batch dim
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=0).unsqueeze(0)  # Add batch dim
        
        # Get multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        
        # Compute logps immediately with the current model state
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
            old_per_token_logps = old_per_token_logps[:, prompt_ids.size(0) - 1:]  # Remove prompt part
            
            # Also compute ref logps if needed
            ref_per_token_logps = None
            if self.beta > 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, **multimodal_inputs
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, input_ids, attention_mask, **multimodal_inputs
                        )
                ref_per_token_logps = ref_per_token_logps[:, prompt_ids.size(0) - 1:]
        
        # Store all data needed for loss computation
        logps_data = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps.squeeze(0),  # Remove batch dim
            "ref_per_token_logps": ref_per_token_logps.squeeze(0) if ref_per_token_logps is not None else None,
            "multimodal_cache": {k: multimodal_inputs[k].squeeze(0) if k in multimodal_inputs and multimodal_inputs[k] is not None else None 
                               for k in ["pixel_values", "image_flags", "image_grid_thw"]}
        }
        
        return logps_data
    
    def _generate_single_completion_with_logps(self, input_item: dict, model, requires_grad=True) -> tuple[str, dict, dict]:
        """Generate completion and compute logps in one forward pass"""
        # First generate the completion
        completion, multimodal_cache = self._generate_single_completion(input_item, model, requires_grad)
        
        # Then compute logps for the generated completion
        logps_data = self._compute_and_cache_logps_for_sample(input_item, completion, model, self.accelerator.device)
        
        return completion, multimodal_cache, logps_data
    
    def _format_two_phase_results_with_precomputed_logps(self, inputs: list[dict], completions: list[str],
                                                       multimodal_caches: list[dict], rewards: list[float], 
                                                       precomputed_logps: list[dict], device) -> dict:
        """Format two-phase results using precomputed logps (avoid second forward pass)"""
        batch_size = len(inputs)
        
        # Use precomputed data to avoid second forward pass
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        all_multimodal_inputs = {"pixel_values": [], "image_flags": [], "image_grid_thw": []}
        
        for i, logps_data in enumerate(precomputed_logps):
            all_prompt_ids.append(logps_data["prompt_ids"])
            all_prompt_masks.append(logps_data["prompt_mask"])
            all_completion_ids.append(logps_data["completion_ids"])
            all_completion_masks.append(logps_data["completion_mask"])
            all_old_per_token_logps.append(logps_data["old_per_token_logps"])
            
            if logps_data["ref_per_token_logps"] is not None:
                all_ref_per_token_logps.append(logps_data["ref_per_token_logps"])
            
            # Use precomputed multimodal data
            for key in ["pixel_values", "image_flags", "image_grid_thw"]:
                if key in logps_data["multimodal_cache"] and logps_data["multimodal_cache"][key] is not None:
                    all_multimodal_inputs[key].append(logps_data["multimodal_cache"][key])
        
        # Pad all sequences to the same length
        max_prompt_len = max(len(ids) for ids in all_prompt_ids)
        max_completion_len = max(len(ids) for ids in all_completion_ids)
        
        # Get tokenizer
        if hasattr(self.processing_class, 'tokenizer'):
            tokenizer = self.processing_class.tokenizer
        else:
            tokenizer = self.processing_class
        pad_token_id = tokenizer.pad_token_id
        
        # Pad sequences and stack tensors (similar to _format_group_results_with_cache)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for prompt_ids, prompt_mask in zip(all_prompt_ids, all_prompt_masks):
            padding_length = max_prompt_len - len(prompt_ids)
            if padding_length > 0:
                padded_ids = torch.cat([torch.full((padding_length,), pad_token_id, dtype=prompt_ids.dtype, device=device), prompt_ids])
                padded_mask = torch.cat([torch.zeros(padding_length, dtype=prompt_mask.dtype, device=device), prompt_mask])
            else:
                padded_ids = prompt_ids
                padded_mask = prompt_mask
            padded_prompt_ids.append(padded_ids)
            padded_prompt_masks.append(padded_mask)
        
        padded_completion_ids = []
        padded_completion_masks = []
        padded_old_logps = []
        for completion_ids, completion_mask, old_logps in zip(all_completion_ids, all_completion_masks, all_old_per_token_logps):
            padding_length = max_completion_len - len(completion_ids)
            if padding_length > 0:
                padded_ids = torch.cat([completion_ids, torch.full((padding_length,), pad_token_id, dtype=completion_ids.dtype, device=device)])
                padded_mask = torch.cat([completion_mask, torch.zeros(padding_length, dtype=completion_mask.dtype, device=device)])
                padded_logps = torch.cat([old_logps, torch.zeros(padding_length, dtype=old_logps.dtype, device=device)])
            else:
                padded_ids = completion_ids
                padded_mask = completion_mask
                padded_logps = old_logps
            padded_completion_ids.append(padded_ids)
            padded_completion_masks.append(padded_mask)
            padded_old_logps.append(padded_logps)
        
        # Stack all tensors
        prompt_ids_tensor = torch.stack(padded_prompt_ids)
        prompt_mask_tensor = torch.stack(padded_prompt_masks)
        completion_ids_tensor = torch.stack(padded_completion_ids)
        completion_mask_tensor = torch.stack(padded_completion_masks)
        old_per_token_logps_tensor = torch.stack(padded_old_logps)
        
        # Handle ref logps if available
        ref_per_token_logps_tensor = None
        if all_ref_per_token_logps:
            padded_ref_logps = []
            for ref_logps in all_ref_per_token_logps:
                padding_length = max_completion_len - len(ref_logps)
                if padding_length > 0:
                    padded_ref = torch.cat([ref_logps, torch.zeros(padding_length, dtype=ref_logps.dtype, device=device)])
                else:
                    padded_ref = ref_logps
                padded_ref_logps.append(padded_ref)
            ref_per_token_logps_tensor = torch.stack(padded_ref_logps)
        
        # Handle multimodal inputs (similar to _format_group_results_with_cache)
        multimodal_inputs = {}
        if all_multimodal_inputs["pixel_values"]:
            try:
                stacked = torch.stack(all_multimodal_inputs["pixel_values"], dim=0)
                if len(stacked.shape) == 5:  # [batch, num_patches, C, H, W]
                    batch_size, num_patches, c, h, w = stacked.shape
                    multimodal_inputs["pixel_values"] = stacked.view(batch_size * num_patches, c, h, w)
                else:
                    multimodal_inputs["pixel_values"] = stacked
            except Exception as e:
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] pixel_values stacking failed: {e}, using dummy tensor")
                batch_size = len(all_multimodal_inputs["pixel_values"])
                dummy_tensor = torch.zeros((batch_size * 7, 3, 448, 448), dtype=torch.bfloat16, device=device)
                multimodal_inputs["pixel_values"] = dummy_tensor
        
        if all_multimodal_inputs["image_flags"]:
            try:
                stacked_flags = torch.stack(all_multimodal_inputs["image_flags"], dim=0)
                if len(stacked_flags.shape) == 2:  # [batch, num_patches]
                    batch_size, num_patches = stacked_flags.shape
                    multimodal_inputs["image_flags"] = stacked_flags.view(batch_size * num_patches)
                else:
                    multimodal_inputs["image_flags"] = stacked_flags
            except Exception as e:
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] image_flags stacking failed: {e}, using concatenation")
                multimodal_inputs["image_flags"] = torch.cat(all_multimodal_inputs["image_flags"], dim=0)
        
        if all_multimodal_inputs["image_grid_thw"]:
            try:
                stacked_grid_thw = torch.stack(all_multimodal_inputs["image_grid_thw"], dim=0)
                multimodal_inputs["image_grid_thw"] = stacked_grid_thw
            except Exception as e:
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] image_grid_thw stacking failed: {e}, using concatenation")
                multimodal_inputs["image_grid_thw"] = torch.cat(all_multimodal_inputs["image_grid_thw"], dim=0)
        
        # Convert rewards to advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = rewards_tensor  # Use raw rewards as advantages for CLEVR
        
        # Log metrics
        completion_length = completion_mask_tensor.sum(dim=1).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(rewards_tensor.mean().item())
        self._metrics["reward_std"].append(rewards_tensor.std().item())
        
        return {
            "prompt_ids": prompt_ids_tensor,
            "prompt_mask": prompt_mask_tensor,
            "completion_ids": completion_ids_tensor,
            "completion_mask": completion_mask_tensor,
            "old_per_token_logps": old_per_token_logps_tensor,  # Precomputed, no second forward pass!
            "ref_per_token_logps": ref_per_token_logps_tensor,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "game_rewards": rewards,
            "completions": completions,
            "inputs": inputs
        }
    
    def _build_integrated_clues_text(self, all_player_clues: list, player_thinking_history: list, current_player_id: int) -> str:
        """Build integrated clues text that includes current player's thinking history"""
        if not all_player_clues:
            return ""
        
        # Parse thinking history by round
        thinking_by_round = {}
        for thinking_entry in player_thinking_history:
            if thinking_entry.startswith("Round "):
                round_num_str = thinking_entry.split(":")[0].strip()
                thinking_text = thinking_entry.split(":", 1)[1].strip()
                thinking_by_round[round_num_str] = thinking_text
        
        # Organize clues by round
        clues_by_round = {}
        for clue_entry in all_player_clues:
            if clue_entry.startswith("Round "):
                # Format: 'Round 1 - Player 1: "clue text"'
                # First split by " - " to separate round from player:clue
                parts = clue_entry.split(" - ", 1)
                if len(parts) >= 2:
                    round_info = parts[0]  # "Round X"
                    player_clue_part = parts[1]  # "Player Y: clue_text"
                    
                    # Then split by ": " to separate player from clue
                    if ": " in player_clue_part:
                        player_info, clue_text = player_clue_part.split(": ", 1)
                        
                        if round_info not in clues_by_round:
                            clues_by_round[round_info] = []
                        clues_by_round[round_info].append((player_info, clue_text))
        
        # Build integrated text
        integrated_lines = []
        for round_key in sorted(clues_by_round.keys()):
            integrated_lines.append(f"{round_key}:")
            for player_info, clue_text in clues_by_round[round_key]:
                if player_info == f"Player {current_player_id}" and round_key in thinking_by_round:
                    # This is current player's clue, add their thinking first, then clue
                    integrated_lines.append(f"{player_info} (thinking): {thinking_by_round[round_key]}")
                    integrated_lines.append(f"{player_info} (clue): {clue_text}")
                else:
                    # Other player's clue, just add the clue
                    integrated_lines.append(f"{player_info}: {clue_text}")
            
            integrated_lines.append("")  # Empty line between rounds
        
        result = "\n".join(integrated_lines).strip()
        return result
    
    def _build_integrated_clues_text_with_own_thinking(self, all_player_clues: list, player_thinking_history: list, current_player_id: int) -> str:
        """Build integrated clues text that includes current player's thinking history from previous rounds
        CRITICAL: Other players can only see ANSWER parts, not thinking!"""
        if not all_player_clues and not player_thinking_history:
            return ""
        
        # Parse thinking history by round
        thinking_by_round = {}
        for thinking_entry in player_thinking_history:
            if thinking_entry.startswith("Round "):
                round_num_str = thinking_entry.split(":")[0].strip()
                thinking_text = thinking_entry.split(":", 1)[1].strip()
                thinking_by_round[round_num_str] = thinking_text
        
        # Organize clues by round - extract only answer content to ensure privacy
        clues_by_round = {}
        for clue_entry in all_player_clues:
            if clue_entry.startswith("Round "):
                # Format: 'Round 1 - Player 1: "clue text"'
                # First split by " - " to separate round from player:clue
                parts = clue_entry.split(" - ", 1)
                if len(parts) >= 2:
                    round_info = parts[0]  # "Round X"
                    player_clue_part = parts[1]  # "Player Y: clue_text"
                    
                    # Then split by ": " to separate player from clue
                    if ": " in player_clue_part:
                        player_info, clue_text = player_clue_part.split(": ", 1)
                        
                        # CRITICAL FIX: Extract only answer content if clue_text contains full response
                        clean_clue_text = self._extract_answer_only_from_text(clue_text)
                        
                        if round_info not in clues_by_round:
                            clues_by_round[round_info] = []
                        clues_by_round[round_info].append((player_info, clean_clue_text))
        
        # Build integrated text showing clues with own thinking appearing before own clue
        integrated_lines = []
        
        # Determine the rounds we have data for
        all_rounds = set()
        for round_key in clues_by_round.keys():
            all_rounds.add(round_key)
        for round_key in thinking_by_round.keys():
            all_rounds.add(round_key)
        
        for round_key in sorted(all_rounds):
            integrated_lines.append(f"{round_key}:")
            
            # Show all clues for this round, but insert current player's thinking before their clue
            if round_key in clues_by_round:
                for player_info, clue_text in clues_by_round[round_key]:
                    # Check if this is the current player's clue
                    if player_info == f"Player {current_player_id}":
                        # Add thinking before the clue (only current player sees their own thinking)
                        if round_key in thinking_by_round:
                            # Use the new format: direct thinking content + boxed answer
                            integrated_lines.append(f"Player {current_player_id}: {thinking_by_round[round_key]}")
                            integrated_lines.append(f"\\boxed{{{clue_text}}}")
                        else:
                            # No thinking available, just show the answer in boxed format
                            integrated_lines.append(f"Player {current_player_id}: \\boxed{{{clue_text}}}")
                    else:
                        # Other players' clues: only show answer content (no thinking)
                        integrated_lines.append(f"{player_info}: {clue_text}")
            
            integrated_lines.append("")  # Empty line between rounds
        
        result = "\n".join(integrated_lines).strip()
        return result
    
    def _extract_answer_only_from_text(self, text: str) -> str:
        """Extract only the answer content from text that might contain thinking tags or boxed format
        This ensures other players cannot see thinking content"""
        import re
        
        # First try to extract from boxed format (new format, handle both single and double backslash)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', text, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # If text appears to be clean (no think/answer tags), return as is
        if '<think>' not in text and '<answer>' not in text and '\\boxed{' not in text:
            return text
        
        # Try to extract from <answer> tags (old format backward compatibility)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Try to find <answer> start and extract content after it (flexible format)
        answer_start_match = re.search(r'<answer>\s*(.*)', text, re.DOTALL)
        if answer_start_match:
            answer_content = answer_start_match.group(1).strip()
            # Clean up any potential closing tags or extra content
            answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
            if answer_content:
                return answer_content
        
        # If no answer tags found, remove thinking tags and return the rest
        # This handles cases where format is broken
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<answer>|</answer>', '', cleaned_text)
        return cleaned_text.strip()
    
    
    def _clean_generated_text(self, text):
        """Clean up repetitive or overly long generated text"""
        import re
        
        if not text or len(text.strip()) == 0:
            return text
            
        # 1. Detect and truncate at excessive repetition
        # Look for patterns where the same words repeat many times
        words = text.split()
        if len(words) > 50:  # Only check long outputs
            # Check for repeating sequences of 3+ words
            for seq_len in range(3, min(10, len(words)//3)):
                for start_idx in range(len(words) - seq_len * 3):
                    seq = words[start_idx:start_idx + seq_len]
                    # Check if this sequence repeats 3+ times
                    repeat_count = 1
                    check_idx = start_idx + seq_len
                    while check_idx + seq_len <= len(words):
                        if words[check_idx:check_idx + seq_len] == seq:
                            repeat_count += 1
                            check_idx += seq_len
                        else:
                            break
                    
                    if repeat_count >= 3:  # Found excessive repetition
                        # Truncate at the start of repetition
                        truncated_words = words[:start_idx + seq_len]
                        text = ' '.join(truncated_words)
                        break
                else:
                    continue
                break
        
        # 2. Truncate if still too long (more than 800 characters)
        # if len(text) > 800:
        #     # Try to truncate at sentence boundary
        #     sentences = re.split(r'[.!?]+', text[:800])
        #     if len(sentences) > 1:
        #         text = '.'.join(sentences[:-1]) + '.'
        #     else:
        #         text = text[:800] + '...'
        
        # 3. Ensure proper format structure is maintained
        # If we have opening tags but no closing tags, try to fix
        if '<think>' in text and text.count('<think>') > text.count('</think>'):
            # Add missing closing tag
            if '</think>' not in text:
                text += '</think>'
        
        if '<answer>' in text and text.count('<answer>') > text.count('</answer>'):
            # Add missing closing tag
            if '</answer>' not in text:
                text += '</answer>'
                
        return text
    
    
    
    def _generate_single_completion(self, input_item: dict, model, requires_grad=True) -> tuple[str, dict]:
        """Generate a single completion for one player's decision"""
        device = self.accelerator.device
        
        # Prepare inputs
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [input_item])
        
        # Handle image loading
        if "image_path" in input_item and input_item["image_path"] is not None:
            images = [PIL.Image.open(p) for p in self._get_key_from_inputs(input_item, "image_path")]
        else:
            images = []

        for img in images:
            try:
                # Ensure minimum dimensions
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
#===========================================================================       
        if self.accelerator.process_index == 0:
            player_id = input_item.get("player_id", "unknown")
            image_paths = input_item.get("image_path", [])
            phase = input_item.get("phase", "unknown")
            round_num = input_item.get("round_num", "unknown")
            
            # Display appropriate header based on phase
            if phase == "clue":
                print(f"\n[CLUE INPUT]  Player {player_id} Round {round_num}")
            elif phase == "decision":
                print(f"\n[DECISION INPUT]  Player {player_id}")
            else:
                print(f"\n[UNKNOWN INPUT]  Player {player_id}")
                
            print("Available keys in input_item:", list(input_item.keys()))
            if 'problem' in input_item:
                print("Raw prompt text  ↓↓↓")
                print(input_item['problem'])
            else:
                print("No 'problem' key found!")
                if 'prompt' in input_item:
                    print("Found 'prompt' key instead:")
                    prompt_content = input_item['prompt']
                    if isinstance(prompt_content, list) and len(prompt_content) > 0:
                        user_content = prompt_content[0].get('content', [])
                        for item in user_content:
                            if item.get('type') == 'text':
                                print("Text content:", item.get('text', ''))
                                break
            print(f"Image path: {image_paths}")
            print()
#===========================================================================
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move to device and handle dtypes
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        
        # Ensure dtypes match the model
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'pixel_values' in prompt_inputs:
                prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'input_ids' in prompt_inputs:
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(dtype=torch.long)
            if 'attention_mask' in prompt_inputs:
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(dtype=torch.long)
        
        # Generate completion based on requires_grad
        if requires_grad:
            # Normal training mode - gradients will be computed
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Create decision-specific generation config to ensure enough tokens for reasoning
                decision_generation_config = GenerationConfig(
                    max_new_tokens=1024,  # Increased for longer reasoning sequences
                    do_sample=True,
                    temperature=self.generation_config.temperature,
                    min_new_tokens=250,   # Increased to force complete answer generation
                    repetition_penalty=1.02,  # Significantly reduced to avoid early stopping
                    pad_token_id=self.generation_config.pad_token_id,
                    eos_token_id=None,  # Disable EOS token to prevent early stopping
                    max_length=None,
                    early_stopping=False,  # Allow full reasoning completion
                    no_repeat_ngram_size=0,  # Disable ngram repetition detection
                )
                # Commented out to keep eos_token_id as None to prevent early stopping
                # if hasattr(self.vlm_module, "get_eos_token_id"):
                #     decision_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                
                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=decision_generation_config,
                    logits_processor=self._rollout_logits_processor(),
                )

                prompt_length = prompt_inputs["input_ids"].size(1)
                if not self.vlm_module.is_embeds_input():
                    completion_ids = generate_returned_result[:, prompt_length:]
                else:
                    completion_ids = generate_returned_result
        else:
            # Inference mode - no gradients needed
            with torch.no_grad():
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    # Create decision-specific generation config to ensure enough tokens for reasoning
                    decision_generation_config = GenerationConfig(
                        max_new_tokens=1024,  # Increased for longer reasoning sequences
                        do_sample=True,
                        temperature=self.generation_config.temperature,
                        min_new_tokens=250,   # Increased to force complete answer generation
                        repetition_penalty=1.02,  # Significantly reduced to avoid early stopping
                        pad_token_id=self.generation_config.pad_token_id,
                        eos_token_id=None,  # Disable EOS token to prevent early stopping
                        max_length=None,
                        early_stopping=False,  # Allow full reasoning completion
                        no_repeat_ngram_size=0,  # Disable ngram repetition detection
                    )
                    if hasattr(self.vlm_module, "get_eos_token_id"):
                        decision_generation_config.eos_token_id = self.vlm_module.get_eos_token_id(self.processing_class)
                    
                    generate_returned_result = unwrapped_model.generate(
                        **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                        generation_config=decision_generation_config,
                        logits_processor=self._rollout_logits_processor(),
                    )

                    prompt_length = prompt_inputs["input_ids"].size(1)
                    if not self.vlm_module.is_embeds_input():
                        completion_ids = generate_returned_result[:, prompt_length:]
                    else:
                        completion_ids = generate_returned_result
        
        # Decode completion
        completion = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
        
        # Clean up repetitive or overly long outputs (disabled for decision phase to avoid truncation)
        # completion = self._clean_generated_text(completion)

 #===========================================================================       
        # Note: Output printing is handled in the calling methods to avoid duplication
        # (_generate_decision_phase_for_training and _generate_clue_phase_for_training)
#===========================================================================
        
        # Return completion along with processed multimodal inputs to avoid recomputation
        processed_multimodal = {}
        for key in ["pixel_values", "image_flags", "image_grid_thw"]:
            if key in prompt_inputs:
                tensor = prompt_inputs[key].squeeze(0)  # Remove batch dim
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] Caching {key} with shape: {tensor.shape}")
                processed_multimodal[key] = tensor
        
        return completion, processed_multimodal
    
    def _calculate_group_rewards(self, game_data: dict, completions: list[str]) -> list[float]:
        """Calculate rewards for all players in a game based on their votes"""
        from clevr_spotdiff_generator import CLEVRSpotDiffGenerator
        
        if not hasattr(self, '_clevr_generator'):
            self._clevr_generator = CLEVRSpotDiffGenerator()
        
        # Extract votes from all player completions
        player_votes = []
        for completion in completions:
            vote_info = self._clevr_generator.extract_vote_from_decision(completion)
            player_votes.append(vote_info)
            
        if self.accelerator.process_index == 0:
            print(f"[DEBUG] Extracted votes: {player_votes}")
        
        # Calculate rewards based on voting results
        rewards = self._clevr_generator.calculate_game_rewards(game_data, player_votes)
        
        return rewards
    
    def _format_group_results_with_cache(self, inputs: list[dict], completions: list[str],
                                       multimodal_caches: list[dict], rewards: list[float], device) -> dict:
        """Format group results using cached multimodal inputs from generation phase"""
        batch_size = len(inputs)
        
        # Process each input to get real prompt and completion tokens
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []
        all_multimodal_inputs = {"pixel_values": [], "image_flags": [], "image_grid_thw": []}
        
        # Get tokenizer
        if hasattr(self.processing_class, 'tokenizer'):
            tokenizer = self.processing_class.tokenizer
        else:
            tokenizer = self.processing_class
        pad_token_id = tokenizer.pad_token_id
        
        for i, (input_item, completion, multimodal_cache) in enumerate(zip(inputs, completions, multimodal_caches)):
            # Re-prepare the prompt (text part only, faster than full multimodal processing)
            prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [input_item])
            
            # Simple tokenization for prompt (without images)
            prompt_tokens = tokenizer(
                prompts_text[0],  # Get the first (and only) prompt
                return_tensors="pt",
                add_special_tokens=False,
                padding=False
            )
            prompt_ids = prompt_tokens["input_ids"].squeeze(0).to(device)
            prompt_mask = prompt_tokens["attention_mask"].squeeze(0).to(device)
            
            # Tokenize the completion
            completion_tokens = tokenizer(
                completion,
                return_tensors="pt",
                add_special_tokens=False,
                padding=False
            )
            completion_ids = completion_tokens["input_ids"].squeeze(0).to(device)
            completion_mask = torch.ones_like(completion_ids, dtype=torch.long, device=device)
            
            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)
            
            # Use cached multimodal inputs (already processed correctly)
            for key in ["pixel_values", "image_flags", "image_grid_thw"]:
                if key in multimodal_cache:
                    cached_tensor = multimodal_cache[key]
                    if self.accelerator.process_index == 0 and i < 3:  # Debug first few samples
                        print(f"[DEBUG] Input {i}, key {key}, cached tensor shape: {cached_tensor.shape}")
                    all_multimodal_inputs[key].append(cached_tensor)
                else:
                    # Fallback for missing data - create tensor with consistent shape
                    if key == "pixel_values":
                        # Create dummy tensor matching InternVL expectations
                        dummy_tensor = torch.zeros((7, 3, 448, 448), dtype=torch.bfloat16, device=device)
                        all_multimodal_inputs[key].append(dummy_tensor)
                        if self.accelerator.process_index == 0:
                            print(f"[DEBUG] Input {i}, key {key}, using dummy tensor shape: {dummy_tensor.shape}")
                    else:
                        dummy_tensor = torch.ones(7, dtype=torch.long, device=device)  # Match num_patches
                        all_multimodal_inputs[key].append(dummy_tensor)
                        if self.accelerator.process_index == 0:
                            print(f"[DEBUG] Input {i}, key {key}, using dummy tensor shape: {dummy_tensor.shape}")
        
        # Pad all sequences to the same length
        max_prompt_len = max(len(ids) for ids in all_prompt_ids)
        max_completion_len = max(len(ids) for ids in all_completion_ids)
        
        # Pad prompt sequences (left padding)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for prompt_ids, prompt_mask in zip(all_prompt_ids, all_prompt_masks):
            padding_length = max_prompt_len - len(prompt_ids)
            if padding_length > 0:
                padded_ids = torch.cat([torch.full((padding_length,), pad_token_id, dtype=prompt_ids.dtype, device=device), prompt_ids])
                padded_mask = torch.cat([torch.zeros(padding_length, dtype=prompt_mask.dtype, device=device), prompt_mask])
            else:
                padded_ids = prompt_ids
                padded_mask = prompt_mask
            padded_prompt_ids.append(padded_ids)
            padded_prompt_masks.append(padded_mask)
        
        # Pad completion sequences (right padding)
        padded_completion_ids = []
        padded_completion_masks = []
        for completion_ids, completion_mask in zip(all_completion_ids, all_completion_masks):
            padding_length = max_completion_len - len(completion_ids)
            if padding_length > 0:
                padded_ids = torch.cat([completion_ids, torch.full((padding_length,), pad_token_id, dtype=completion_ids.dtype, device=device)])
                padded_mask = torch.cat([completion_mask, torch.zeros(padding_length, dtype=completion_mask.dtype, device=device)])
            else:
                padded_ids = completion_ids
                padded_mask = completion_mask
            padded_completion_ids.append(padded_ids)
            padded_completion_masks.append(padded_mask)
        
        # Stack all tensors
        prompt_ids_tensor = torch.stack(padded_prompt_ids)
        prompt_mask_tensor = torch.stack(padded_prompt_masks)
        completion_ids_tensor = torch.stack(padded_completion_ids)
        completion_mask_tensor = torch.stack(padded_completion_masks)
        
        # Stack cached multimodal inputs with InternVL compatibility handling
        multimodal_inputs = {}
        if all_multimodal_inputs["pixel_values"]:
            pixel_values_list = all_multimodal_inputs["pixel_values"]
            if self.accelerator.process_index == 0:
                print(f"[DEBUG] Stacking {len(pixel_values_list)} cached pixel_values")
                for i, pv in enumerate(pixel_values_list[:3]):  # Show first 3 shapes
                    print(f"[DEBUG] Cached pixel_values[{i}] shape: {pv.shape}")
            
            # Handle different tensor shapes for InternVL  
            try:
                 # First, ensure all tensors have the same shape
                 target_shape = pixel_values_list[0].shape
                 consistent_tensors = []
                 
                 for i, pv in enumerate(pixel_values_list):
                     if pv.shape == target_shape:
                         consistent_tensors.append(pv)
                     else:
                         if self.accelerator.process_index == 0:
                             print(f"[DEBUG] Tensor {i} shape mismatch: {pv.shape} vs target {target_shape}")
                         # Reshape to match target shape if possible
                         if pv.numel() == torch.Size(target_shape).numel():
                             reshaped_pv = pv.view(target_shape)
                             consistent_tensors.append(reshaped_pv)
                             if self.accelerator.process_index == 0:
                                 print(f"[DEBUG] Reshaped tensor {i} to {target_shape}")
                         else:
                             # Create a dummy tensor with target shape
                             dummy_pv = torch.zeros(target_shape, dtype=pv.dtype, device=pv.device)
                             consistent_tensors.append(dummy_pv)
                             if self.accelerator.process_index == 0:
                                 print(f"[DEBUG] Created dummy tensor {i} with shape {target_shape}")
                 
                 # Now stack the consistent tensors
                 stacked = torch.stack(consistent_tensors, dim=0)
                 if self.accelerator.process_index == 0:
                     print(f"[DEBUG] Stack successful, shape: {stacked.shape}")
                 
                 # For InternVL, if we have 5D tensor, flatten batch and patches
                 if len(stacked.shape) == 5:  # [batch, num_patches, C, H, W]
                     batch_size, num_patches, c, h, w = stacked.shape
                     multimodal_inputs["pixel_values"] = stacked.view(batch_size * num_patches, c, h, w)
                 else:  # [batch, C, H, W] or [batch, num_patches*C, H, W]
                     multimodal_inputs["pixel_values"] = stacked
                     
                 if self.accelerator.process_index == 0:
                     print(f"[DEBUG] Final cached pixel_values shape: {multimodal_inputs['pixel_values'].shape}")
                     
            except Exception as e:
                 if self.accelerator.process_index == 0:
                     print(f"[DEBUG] Stack with shape normalization failed: {e}")
                 # Ultimate fallback: create dummy tensor
                 batch_size = len(pixel_values_list)
                 dummy_tensor = torch.zeros((batch_size * 7, 3, 448, 448), dtype=torch.bfloat16, device=device)
                 multimodal_inputs["pixel_values"] = dummy_tensor
                 if self.accelerator.process_index == 0:
                     print(f"[DEBUG] Using ultimate fallback pixel_values shape: {dummy_tensor.shape}")
        
        if all_multimodal_inputs["image_flags"]:
            image_flags_list = all_multimodal_inputs["image_flags"]
            if self.accelerator.process_index == 0:  
                print(f"[DEBUG] Stacking {len(image_flags_list)} cached image_flags")
                for i, img_flag in enumerate(image_flags_list[:3]):  # Show first 3 shapes
                    print(f"[DEBUG] Cached image_flags[{i}] shape: {img_flag.shape}")
            
            try:
                stacked_flags = torch.stack(image_flags_list, dim=0)
                if len(stacked_flags.shape) == 2:  # [batch, num_patches]
                    batch_size, num_patches = stacked_flags.shape
                    multimodal_inputs["image_flags"] = stacked_flags.view(batch_size * num_patches)
                else:  # [batch] 
                    multimodal_inputs["image_flags"] = stacked_flags
                    
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] Final cached image_flags shape: {multimodal_inputs['image_flags'].shape}")
                    
            except Exception as e:
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] image_flags stack failed: {e}, trying concatenation")
                multimodal_inputs["image_flags"] = torch.cat(image_flags_list, dim=0)
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] Concatenated image_flags shape: {multimodal_inputs['image_flags'].shape}")
        
        if all_multimodal_inputs["image_grid_thw"]:
            image_grid_thw_list = all_multimodal_inputs["image_grid_thw"]
            if self.accelerator.process_index == 0:
                print(f"[DEBUG] Stacking {len(image_grid_thw_list)} cached image_grid_thw")
                for i, grid_thw in enumerate(image_grid_thw_list[:3]):  # Show first 3 shapes
                    print(f"[DEBUG] Cached image_grid_thw[{i}] shape: {grid_thw.shape}")
            
            try:
                stacked_grid_thw = torch.stack(image_grid_thw_list, dim=0)
                multimodal_inputs["image_grid_thw"] = stacked_grid_thw
                    
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] Final cached image_grid_thw shape: {multimodal_inputs['image_grid_thw'].shape}")
                    
            except Exception as e:
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] image_grid_thw stack failed: {e}, trying concatenation")
                multimodal_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
                if self.accelerator.process_index == 0:
                    print(f"[DEBUG] Concatenated image_grid_thw shape: {multimodal_inputs['image_grid_thw'].shape}")
        
        # Convert rewards to advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # For CLEVR spot-the-difference games, use raw rewards as advantages
        # since they are already carefully designed for each player's performance
        advantages = rewards_tensor
        
        # **FIX: Pre-compute old_per_token_logps to avoid extra forward pass in compute_loss**
        # Concatenate prompt and completion for full sequence
        input_ids = torch.cat([prompt_ids_tensor, completion_ids_tensor], dim=1)
        attention_mask = torch.cat([prompt_mask_tensor, completion_mask_tensor], dim=1)
        
        # Compute old_per_token_logps with torch.no_grad() to avoid tracking gradients
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(self.model, input_ids, attention_mask, **multimodal_inputs)
            # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
            old_per_token_logps = old_per_token_logps[:, prompt_ids_tensor.size(1) - 1:]
        
        # Also compute ref_per_token_logps if needed
        ref_per_token_logps = None
        if self.beta > 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, **multimodal_inputs
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, **multimodal_inputs
                        )
                # Get rid of the prompt
                ref_per_token_logps = ref_per_token_logps[:, prompt_ids_tensor.size(1) - 1:]
        
        # Log metrics
        completion_length = completion_mask_tensor.sum(dim=1).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(rewards_tensor.mean().item())
        self._metrics["reward_std"].append(rewards_tensor.std().item())
        
        return {
            "prompt_ids": prompt_ids_tensor,
            "prompt_mask": prompt_mask_tensor,
            "completion_ids": completion_ids_tensor,
            "completion_mask": completion_mask_tensor,
            "old_per_token_logps": old_per_token_logps,  # Now pre-computed to avoid extra forward
            "ref_per_token_logps": ref_per_token_logps,  # Pre-computed if needed
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "game_rewards": rewards,
            "completions": completions,
            "inputs": inputs
        }
    


    


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Add debug print to see if training is running (DISABLED for cleaner output)
        # if self.accelerator.process_index == 0:
        #     print(f"[DEBUG] compute_loss called with {len(inputs)} inputs")

        # Simple data handling - now that format is consistent
        actual_inputs = inputs
        
        # Check if this is CLEVR spot-the-difference game which requires two-phase processing
        is_clevr_spotdiff = any(
            item.get("accu_reward_method") == "clevr_spotdiff" for item in actual_inputs
        )
        
        # if self.accelerator.process_index == 0:
        #     print(f"[DEBUG] is_clevr_spotdiff: {is_clevr_spotdiff}")
        
        if is_clevr_spotdiff:
            if self.use_vllm:
                raise ValueError(
                    "The legacy two-phase clevr_spotdiff trainer requires use_vllm=False. "
                    "Use the self_evolve_refined_grpo task path for vLLM GRPO rollouts."
                )
            # Handle CLEVR spot-the-difference two-phase training
            # if self.accelerator.process_index == 0:
            #     print("[DEBUG] Entering CLEVR spot-diff training path")
            processed_inputs = self._handle_clevr_spotdiff_training(actual_inputs, model)
        else:
            # Standard GRPO training
            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] Standard GRPO training, num_generations: {self.num_generations}")
            print(self.num_generations)
            if self.state.global_step % self.num_iterations == 0:
                processed_inputs = self._generate_and_score_completions(actual_inputs, model)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = processed_inputs
            else:
                processed_inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        
        self._step += 1

        # Check if this is CLEVR group-based training
        if "game_rewards" in processed_inputs:
            # CLEVR group-based training - use standard GRPO logic with group rewards
            if self.accelerator.process_index == 0:
                print("[DEBUG] Computing loss for CLEVR group-based training using standard GRPO logic")
            
            # Get training phase for logging
            training_phase = getattr(self.script_args, 'training_phase', 'both')
            
            # All phases now use immediate training - no effective games accumulation
            if self.accelerator.process_index == 0:
                if training_phase == "clue":
                    print(f"[CLUE TRAINING] 🎯 IMMEDIATE TRAINING - Processing batch directly")
                    print(f"[CLUE TRAINING] Processing batch with {len(processed_inputs.get('advantages', []))} samples")
                elif training_phase == "decision":
                    print(f"[DECISION TRAINING] 🎯 IMMEDIATE TRAINING - Processing batch directly")
                    print(f"[DECISION TRAINING] Processing batch with {len(processed_inputs.get('advantages', []))} samples")
                else:  # both
                    print(f"[BOTH PHASES TRAINING] 🎯 IMMEDIATE TRAINING - Processing batch directly")
                    print(f"[BOTH PHASES TRAINING] Processing batch with {len(processed_inputs.get('advantages', []))} samples")
            
            # Normal processing for single batch
            # This applies to:
            # - Clue phase training (always immediate processing)  
            # - Decision phase when min_effective_games <= 1 or threshold reached
            
            # Get training phase information
            training_phase = getattr(self.script_args, 'training_phase', 'both')
            phase_labels = processed_inputs.get("phase_labels", [])
            
            # Filter samples based on training phase
            if training_phase == "clue":
                # Only train on clue phase samples (exclude god_decision_for_clue_reward samples)
                training_mask = [i for i, phase in enumerate(phase_labels) if phase == "clue"]
            elif training_phase == "decision":
                # Only train on decision phase samples (both god_decision and god_decision_for_clue_reward)
                training_mask = [i for i, phase in enumerate(phase_labels) if phase in ["god_decision", "god_decision_for_clue_reward"]]
            else:  # "both"
                # Train on all samples
                training_mask = list(range(len(phase_labels)))
            
            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] Training phase: {training_phase}")
            #     print(f"[DEBUG] Phase labels: {phase_labels}")
            #     print(f"[DEBUG] Training mask (indices): {training_mask}")
            #     print(f"[DEBUG] Training {len(training_mask)} out of {len(phase_labels)} samples")
            #     
            #     # Count samples by phase
            #     clue_count = len([p for p in phase_labels if p == "clue"])
            #     god_decision_count = len([p for p in phase_labels if p in ["god_decision", "god_decision_for_clue_reward"]])
            #     print(f"[DEBUG] Sample breakdown: {clue_count} clue, {god_decision_count} god_decision (including for clue reward)")
                
                if training_phase == "clue":
                    print(f"[CLUE TRAINING] 🎯 Training ONLY clue phase - Decision samples included for reward calculation only")
                    print(f"[CLUE TRAINING] Direct batch processing - immediate parameter updates")
                elif training_phase == "decision":
                    print(f"[DECISION TRAINING] 🎯 Training ONLY decision phase - Clue phase was inference only")
                    print(f"[DECISION TRAINING] Direct batch processing - immediate parameter updates")
                elif training_phase == "both":
                    print(f"[BOTH PHASES] 🎯 Training both clue and decision phases simultaneously")
                    print(f"[BOTH PHASES] Direct batch processing - immediate parameter updates")
            
            # If no samples to train on, return zero loss
            if not training_mask:
                if self.accelerator.process_index == 0:
                    print("[WARNING] No samples to train on for current phase, returning zero loss")
                return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            
            # Get the prepared inputs (now they are real tokens, not dummy)
            prompt_ids, prompt_mask = processed_inputs["prompt_ids"], processed_inputs["prompt_mask"]
            completion_ids, completion_mask = processed_inputs["completion_ids"], processed_inputs["completion_mask"]
            multimodal_inputs = processed_inputs["multimodal_inputs"]
            
            # Filter tensors to only include training samples
            prompt_ids = prompt_ids[training_mask]
            prompt_mask = prompt_mask[training_mask]
            completion_ids = completion_ids[training_mask]
            completion_mask = completion_mask[training_mask]
            
            # Filter multimodal inputs if they exist
            if "pixel_values" in multimodal_inputs and multimodal_inputs["pixel_values"] is not None:
                # Handle InternVL format where pixel_values might be flattened
                pixel_values = multimodal_inputs["pixel_values"]
                if pixel_values.dim() == 4:  # [batch*patches, C, H, W]
                    # Determine how many patches per sample
                    patches_per_sample = pixel_values.size(0) // len(phase_labels)
                    # Reshape to [batch, patches, C, H, W] for easier indexing
                    pixel_values = pixel_values.view(len(phase_labels), patches_per_sample, *pixel_values.shape[1:])
                    # Filter and flatten back
                    filtered_pixel_values = pixel_values[training_mask]
                    multimodal_inputs["pixel_values"] = filtered_pixel_values.view(-1, *pixel_values.shape[2:])
                else:
                    multimodal_inputs["pixel_values"] = pixel_values[training_mask]
            
            if "image_flags" in multimodal_inputs and multimodal_inputs["image_flags"] is not None:
                image_flags = multimodal_inputs["image_flags"]
                if image_flags.dim() == 1:  # [batch*patches]
                    # Determine how many patches per sample
                    patches_per_sample = image_flags.size(0) // len(phase_labels)
                    # Reshape to [batch, patches] for easier indexing
                    image_flags = image_flags.view(len(phase_labels), patches_per_sample)
                    # Filter and flatten back
                    filtered_image_flags = image_flags[training_mask]
                    multimodal_inputs["image_flags"] = filtered_image_flags.view(-1)
                else:
                    multimodal_inputs["image_flags"] = image_flags[training_mask]
            
            if "image_grid_thw" in multimodal_inputs and multimodal_inputs["image_grid_thw"] is not None:
                image_grid_thw = multimodal_inputs["image_grid_thw"]
                multimodal_inputs["image_grid_thw"] = image_grid_thw[training_mask]
            
            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] Filtered prompt_ids shape: {prompt_ids.shape}")
            #     print(f"[DEBUG] Filtered completion_ids shape: {completion_ids.shape}")
            #     if "pixel_values" in multimodal_inputs and multimodal_inputs["pixel_values"] is not None:
            #         print(f"[DEBUG] Filtered pixel_values shape: {multimodal_inputs['pixel_values'].shape}")
            #     if "image_flags" in multimodal_inputs and multimodal_inputs["image_flags"] is not None:
            #         print(f"[DEBUG] Filtered image_flags shape: {multimodal_inputs['image_flags'].shape}")
            
            # Concatenate for full sequence
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # Get the current policy's log probabilities
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
            # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
            per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

            # Get the advantages from inputs (also filter them)
            advantages = processed_inputs["advantages"][training_mask]

            # **FIX: Use pre-computed old_per_token_logps instead of current logps**
            # Check if old_per_token_logps was pre-computed, otherwise fall back to detached current logps
            old_per_token_logps = processed_inputs.get("old_per_token_logps")
            if old_per_token_logps is not None:
                old_per_token_logps = old_per_token_logps[training_mask]
                # if self.accelerator.process_index == 0:
                #     print("[DEBUG] Using pre-computed filtered old_per_token_logps")
            else:
                # Fallback: use current logps (effectively REINFORCE without ratio clipping)
                old_per_token_logps = per_token_logps.detach()
                # if self.accelerator.process_index == 0:
                #     print("[DEBUG] Using fallback: current logps as old_per_token_logps")

            # Compute the policy ratio and clipped version.
            # Clamp the log-ratio BEFORE exp() (verl-style). This is the root-cause guard for the
            # intermittent grad_norm=nan: an uncapped exp() can overflow to inf on extreme tokens
            # (bf16 rollout vs. recompute numerical mismatch), and backward through an inf node
            # yields inf/nan gradients regardless of any downstream nan_to_num on the loss *value*.
            # clamp(-10, 10) bounds coef_1 to [e^-10, e^10] and zeroes the gradient outside the range.
            log_ratio = (per_token_logps - old_per_token_logps).clamp(-10, 10)
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # Add KL penalty if beta > 0 and ref_per_token_logps available
            if self.beta > 0:
                ref_per_token_logps = processed_inputs.get("ref_per_token_logps")
                if ref_per_token_logps is not None:
                    ref_per_token_logps = ref_per_token_logps[training_mask]
                    # Clamp exponent to keep the k3 KL term finite (see standard-GRPO path).
                    delta = (ref_per_token_logps - per_token_logps).clamp(-20, 20)
                    per_token_kl = torch.exp(delta) - delta - 1
                    per_token_loss = per_token_loss + self.beta * per_token_kl
                    
                    # Log KL divergence. clamp(min=1.0) guards 0/0 on fully-padded rows
                    # (this path builds mask from ones_like + zero padding, so empty rows are possible).
                    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
                    self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
                else:
                    # No reference model logps available
                    self._metrics["kl"].append(0.0)

            # Compute final loss. nan_to_num + clamp mirror the standard-GRPO path guards.
            per_token_loss = torch.nan_to_num(per_token_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()

            # Log clip ratio
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
            
            # Note: Reward metrics are already logged in _calculate_two_phase_rewards with original values
            # Don't log them again here to avoid overwriting
            
            # if self.accelerator.process_index == 0:
            #     print(f"[DEBUG] CLEVR {training_phase} phase loss: {loss.item()}, mean advantage: {advantages.mean().item()}")
            #     print(f"[DEBUG] Mean per-token loss: {per_token_loss.mean().item()}")
            
            # Additional training phase specific information
            if training_phase == "clue":
                print(f"[CLUE TRAINING] ✅ Successfully computed clue phase loss")
                print(f"[CLUE TRAINING] Optimizing for content quality and format")
            elif training_phase == "decision": 
                print(f"[DECISION TRAINING] ✅ Successfully computed decision phase loss")
                print(f"[DECISION TRAINING] Optimizing for voting accuracy and format")
            
            # Log phase-specific metrics
            self._metrics[f"loss_{training_phase}_phase"].append(loss.item())
            self._metrics[f"advantages_{training_phase}_phase"].append(advantages.mean().item())
        
        else:
            # Standard GRPO training
            # Get the prepared inputs
            prompt_ids, prompt_mask = processed_inputs["prompt_ids"], processed_inputs["prompt_mask"]
            completion_ids, completion_mask = processed_inputs["completion_ids"], processed_inputs["completion_mask"]
            multimodal_inputs = processed_inputs["multimodal_inputs"]
            
            # Concatenate for full sequence
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # Get the current policy's log probabilities
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
            # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
            per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

            # Get the advantages from inputs
            advantages = processed_inputs["advantages"]

            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
            # and use per_token_logps.detach() instead
            old_per_token_logps = processed_inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

            # Compute the policy ratio and clipped version.
            # Clamp the log-ratio BEFORE exp() (verl-style) — see the matching guard in the other
            # loss branch above for the full rationale. Bounds coef_1 to [e^-10, e^10] so an extreme
            # token cannot push exp() to inf and poison the backward graph with inf/nan gradients.
            log_ratio = (per_token_logps - old_per_token_logps).clamp(-10, 10)
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # Add KL penalty if beta > 0
            if self.beta > 0:
                ref_per_token_logps = processed_inputs["ref_per_token_logps"]
                # k3 KL estimator goes into the loss (not just logging), so its bare exp() is a
                # live nan/inf source: when policy logp << ref logp, (ref - policy) is large and
                # exp() overflows to inf -> loss=inf -> grad_norm=nan. Clamp the exponent to keep
                # it finite (exp(20)~=4.8e8) while preserving gradient flow.
                delta = (ref_per_token_logps - per_token_logps).clamp(-20, 20)
                per_token_kl = torch.exp(delta) - delta - 1
                per_token_loss = per_token_loss + self.beta * per_token_kl

                # Log KL divergence. clamp(min=1.0) guards the 0/0 that occurs when a rollout
                # is fully masked (empty completion): completion_mask.sum==0 -> nan.
                mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
                self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

            # Compute final loss. Two guards against the intermittent grad_norm=nan that then
            # poisons the weights and crashes the next rollout with "probability tensor contains
            # inf/nan": (1) nan_to_num scrubs any non-finite per-token loss so one bad sample
            # cannot poison the whole batch; (2) clamp(min=1.0) on the per-sequence token count
            # avoids the 0/0 when a completion is empty / fully masked.
            per_token_loss = torch.nan_to_num(per_token_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()

            # Log clip ratio
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        # A-2 grad-finite gate (verl-style skip). Under DeepSpeed ZeRO-3 the optimizer step runs
        # *inside* accelerator.backward(), so HF's post-backward grad-norm hook is too late to skip.
        # The correct interception is here, on the loss that feeds backward: if it is non-finite,
        # detach the poisoned graph and return a fresh leaf zero-tensor (identical to the "no samples"
        # skip at the top of this method). backward then produces finite zero-gradients -> the engine
        # step is a no-op, so a stray nan never reaches the weights (which is what crashes the next
        # rollout). Each rank checks its own scalar with identical control flow, so ZeRO-3 collectives
        # stay in lock-step and cannot hang. A-1's log-ratio clamp already keeps the graph finite, so
        # this is the last-resort net and should rarely fire.
        if loss is not None and not torch.isfinite(loss):
            if self.accelerator.is_main_process:
                print(
                    f"[GRAD-SKIP] step={self.state.global_step} non-finite loss={loss.item()} "
                    f"-> skipping update (returning zero loss)"
                )
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Optional[Sampler]:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        # For IterableDataset, we don't need a sampler
        if isinstance(self.train_dataset, (IterableDataset, DynamicIterableDataset, EpochAwareIterableDataset, CyclicDynamicDataset)):
            return None
            
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        # Use CyclicRepeatSampler to ensure same samples within num_iterations cycle
        return CyclicRepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            cycle_length=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Optional[Sampler]:
        """Returns a sampler for evaluation."""
        # For IterableDataset, we don't need a sampler
        if isinstance(eval_dataset, (IterableDataset, DynamicIterableDataset, EpochAwareIterableDataset, CyclicDynamicDataset)):
            return None
        
        effective_batch_size = (
            self.args.per_device_eval_batch_size
            * self.accelerator.num_processes
        )
        
        # For evaluation, we use CyclicRepeatSampler with cycle_length=1 (no need for cycles in eval)
        return CyclicRepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            cycle_length=1,
            seed=self.args.seed,
        )
    
    
    def get_train_dataloader(self):
        """Override to inject epoch tracking into dataloader for dynamic datasets."""
        dataloader = super().get_train_dataloader()
        
        # If we have a dynamic dataset, add simple epoch setting
        if isinstance(self.train_dataset, (DynamicIterableDataset, EpochAwareIterableDataset, CyclicDynamicDataset)):
            # Store original set_epoch method if it exists
            original_set_epoch = getattr(dataloader, 'set_epoch', None)
            
            def set_epoch(epoch):
                print(f"Setting epoch {epoch} on dynamic dataset")
                # Call original set_epoch if it exists
                if original_set_epoch is not None:
                    original_set_epoch(epoch)
                # Set epoch on our dynamic dataset
                if hasattr(self.train_dataset, 'set_epoch'):
                    self.train_dataset.set_epoch(epoch)
            
            # Monkey patch the set_epoch method
            dataloader.set_epoch = set_epoch
        
        return dataloader

    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to add S3 sync after each checkpoint save"""
        # Call parent class save method
        result = super()._save_checkpoint(model, trial)
        
        # Execute S3 sync command after successful checkpoint save
        try:
            import subprocess
            sync_command = [
                "aws", "s3", "sync", 
                "/mnt/localssd/output", 
                "s3://qinsiwang/0814-InternVL-Interactive-now"
            ]
            
            if self.accelerator.process_index == 0:  # Only sync from main process
                print(f"[S3 SYNC] Starting sync to S3...")
                result_sync = subprocess.run(
                    sync_command, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                print(f"[S3 SYNC] ✅ Successfully synced checkpoint to S3")
                print(f"[S3 SYNC] Output: {result_sync.stdout}")
                
        except subprocess.CalledProcessError as e:
            print(f"[S3 SYNC] ❌ Failed to sync to S3: {e}")
            print(f"[S3 SYNC] Error output: {e.stderr}")
        except FileNotFoundError:
            print(f"[S3 SYNC] ❌ AWS CLI not found. Please install AWS CLI to enable S3 sync.")
        except Exception as e:
            print(f"[S3 SYNC] ❌ Unexpected error during S3 sync: {e}")
        
        return result
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override to add S3 sync after final model save"""
        # Call parent class save method
        result = super().save_model(output_dir, _internal_call)
        
        # Execute S3 sync command after final model save
        if not _internal_call:  # Only sync for final save, not internal checkpoint saves
            try:
                import subprocess
                sync_command = [
                    "aws", "s3", "sync", 
                    "/mnt/localssd/output", 
                    "s3://qinsiwang/0814-InternVL-Interactive-now"
                ]
                
                if self.accelerator.process_index == 0:  # Only sync from main process
                    print(f"[S3 SYNC] Starting final model sync to S3...")
                    result_sync = subprocess.run(
                        sync_command, 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    print(f"[S3 SYNC] ✅ Successfully synced final model to S3")
                    print(f"[S3 SYNC] Output: {result_sync.stdout}")
                    
            except subprocess.CalledProcessError as e:
                print(f"[S3 SYNC] ❌ Failed to sync final model to S3: {e}")
                print(f"[S3 SYNC] Error output: {e.stderr}")
            except FileNotFoundError:
                print(f"[S3 SYNC] ❌ AWS CLI not found. Please install AWS CLI to enable S3 sync.")
            except Exception as e:
                print(f"[S3 SYNC] ❌ Unexpected error during final model S3 sync: {e}")
        
        return result
