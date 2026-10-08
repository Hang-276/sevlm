from typing import Callable, Optional, Any, Dict, Union
from copy import deepcopy
from contextlib import contextmanager
import inspect
import torch
import numpy as np
from torch.utils.data import IterableDataset
from transformers.utils import logging
import random

from open_r1.grpo_data import prepare_grpo_sample

logger = logging.get_logger(__name__)


class DynamicIterableDataset(IterableDataset):
    """
    A dynamic iterable dataset that generates new data for each epoch using a data generation function.
    """
    
    def __init__(
        self,
        data_generator_func: Callable[[], Dict[str, Any]],
        epoch_size: int,
        vlm_module=None,
        question_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_generator_func: A function that generates a single data sample when called
            epoch_size: Number of samples to generate per epoch
            vlm_module: VLM module for processing
            question_prompt: Template for formatting questions
            seed: Random seed for reproducibility
        """
        self.data_generator_func = data_generator_func
        self.epoch_size = epoch_size
        self.vlm_module = vlm_module
        self.question_prompt = question_prompt or "Question: {Question}"
        self.seed = seed
        self.current_epoch = 0
        try:
            inspect.signature(data_generator_func).bind(epoch=0, sample_idx=0)
            self._accepts_epoch_arguments = True
        except (TypeError, ValueError):
            self._accepts_epoch_arguments = False
        
    def set_epoch(self, epoch: int):
        """Set the current epoch (called by trainer)"""
        self.current_epoch = epoch
        logger.info(f"DynamicIterableDataset: Starting epoch {epoch}")

    @contextmanager
    def _generation_rng(self, index):
        if self.seed is None:
            yield
            return
        seed = self.seed + self.current_epoch * self.epoch_size + index
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.random.default_generator.manual_seed(seed)
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def _generate_sample(self, index, with_epoch=False):
        with self._generation_rng(index):
            if with_epoch and self._accepts_epoch_arguments:
                return self.data_generator_func(epoch=self.current_epoch, sample_idx=index)
            return self.data_generator_func()
    
    def __iter__(self):
        """Generate samples for current epoch"""
        for i in range(self.epoch_size):
            try:
                # Generate new data sample
                sample = self._generate_sample(i)
                
                # Process the sample to match expected format
                processed_sample = self._process_sample(sample)
                
                yield processed_sample
                
            except Exception as e:
                logger.warning(f"Failed to generate sample {i} in epoch {self.current_epoch}: {e}")
                continue
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw sample from generator function to match trainer expected format
        
        This must return exactly the same format as make_conversation_from_jsonl in grpo_jsonl.py
        to ensure compatibility with the trainer and accelerate.
        """
        # Handle CLEVR spot-the-difference special case
        if sample.get('accu_reward_method') == 'clevr_spotdiff':
            # CLEVR samples have a different structure - just pass through as-is
            # The trainer handles the two-phase processing directly
            return sample
        
        processed = prepare_grpo_sample(sample, self.question_prompt)
        
        # Handle metadata fields that need to be preserved for reward calculation
        # For sudoku-specific fields, preserve original format; for others, wrap in list
        sudoku_specific_fields = ['puzzle_board', 'solution_board']
        other_metadata_fields = ['sudoku_metadata']
        
        # Handle sudoku-specific fields without extra wrapping
        for field in sudoku_specific_fields:
            if field in sample:
                processed[field] = sample[field]  # Keep original format
        
        # Handle other metadata fields with list wrapping
        for field in other_metadata_fields:
            if field in sample:
                processed[field] = [sample[field]]
        
        return processed
    
    def __len__(self):
        """Return epoch size"""
        return self.epoch_size


class EpochAwareIterableDataset(DynamicIterableDataset):
    """
    Extension that automatically increments epoch and provides epoch info to generator
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch_counter = 0
    
    def __iter__(self):
        """Generate samples with epoch information"""
        successful_samples = 0
        failed_samples = 0
        max_failures = self.epoch_size // 2  # Allow up to 50% failures
        
        for i in range(self.epoch_size):
            try:
                sample = self._generate_sample(i, with_epoch=True)
                
                # Validate that we got a valid sample
                if sample is None:
                    raise ValueError("Generator returned None sample")
                
                processed_sample = self._process_sample(sample)
                yield processed_sample
                successful_samples += 1
                
            except Exception as e:
                failed_samples += 1
                logger.warning(f"Failed to generate sample {i} in epoch {self.current_epoch}: {e}")
                
                # If we have too many failures, stop and report the issue
                if failed_samples > max_failures:
                    logger.error(f"Too many sample generation failures ({failed_samples}/{i+1}). Stopping epoch.")
                    break
                
                continue
        
        logger.info(f"EpochAwareIterableDataset epoch {self.current_epoch}: {successful_samples} successful, {failed_samples} failed samples")


class CyclicDynamicDataset(IterableDataset):
    """
    A wrapper around dynamic datasets that ensures the same samples are repeated
    within each num_iterations cycle for GRPO training, avoiding image regeneration waste.
    """
    
    def __init__(
        self,
        base_dataset: Union[DynamicIterableDataset, EpochAwareIterableDataset],
        num_generations: int,
        num_iterations: int,
        batch_size: int = 1,
    ):
        """
        Args:
            base_dataset: The underlying dynamic dataset
            num_generations: Number of generations per prompt (mini_repeat_count)
            num_iterations: Number of iterations in each cycle (cycle_length)
        """
        self.base_dataset = base_dataset
        self.num_generations = num_generations
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        for name in ('num_generations', 'num_iterations', 'batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self._unique_sample_count() == 0:
            raise ValueError("Dynamic epoch must contain at least one complete GRPO batch")
        self.current_epoch = 0
        self._cached_samples = []
        self._cache_epoch = -1
        
    def set_epoch(self, epoch: int):
        """Set epoch for both this wrapper and the base dataset"""
        self.current_epoch = epoch
        if hasattr(self.base_dataset, 'set_epoch'):
            self.base_dataset.set_epoch(epoch)
        
        # Clear cache when epoch changes to force regeneration
        if epoch != self._cache_epoch:
            self._cached_samples = []
            self._cache_epoch = epoch
            logger.info(f"CyclicDynamicDataset: Starting epoch {epoch}, clearing sample cache")
    
    def _generate_cycle_samples(self):
        """Generate samples for one complete cycle"""
        if not self._cached_samples or self._cache_epoch != self.current_epoch:
            base_iterator = iter(self.base_dataset)
            samples = []
            needed = self._unique_sample_count()
            while len(samples) < needed:
                try:
                    samples.append(next(base_iterator))
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Dynamic generator produced {len(samples)} samples; a complete epoch needs {needed}"
                    ) from exc
            cached = []
            for offset in range(0, needed, self.batch_size):
                batch = samples[offset:offset + self.batch_size]
                for _ in range(self.num_iterations):
                    for sample in batch:
                        cached.extend([sample] * self.num_generations)
            self._cached_samples = cached
            self._cache_epoch = self.current_epoch

    def _unique_sample_count(self):
        unique = len(self.base_dataset) // self.num_generations
        return unique // self.batch_size * self.batch_size
    
    def __iter__(self):
        """Return cached samples in cyclic pattern"""
        self._generate_cycle_samples()
        
        for sample in self._cached_samples:
            yield deepcopy(sample)
    
    def __len__(self):
        """Return total length including repetitions"""
        return self._unique_sample_count() * self.num_generations * self.num_iterations
