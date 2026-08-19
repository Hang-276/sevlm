from typing import Callable, Optional, Any, Dict, Union
import torch
from torch.utils.data import IterableDataset
from transformers.utils import logging
import random

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
        
    def set_epoch(self, epoch: int):
        """Set the current epoch (called by trainer)"""
        self.current_epoch = epoch
        # Set different seed for each epoch to ensure variety
        if self.seed is not None:
            random.seed(self.seed + epoch)
            torch.manual_seed(self.seed + epoch)
        logger.info(f"DynamicIterableDataset: Starting epoch {epoch}")
    
    def __iter__(self):
        """Generate samples for current epoch"""
        for i in range(self.epoch_size):
            try:
                # Generate new data sample
                sample = self.data_generator_func()
                
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
        
        # Extract basic required fields for standard samples
        processed = {
            'problem': sample.get('problem', ''),
            'solution': sample.get('solution', ''),
            'accu_reward_method': sample.get('accu_reward_method', 'default'),
        }
        
        # Ensure solution has proper format
        if not processed['solution'].startswith('<answer>'):
            processed['solution'] = f"<answer> {processed['solution']} </answer>"
        
        # Handle image paths
        if 'image_path' in sample and sample['image_path'] is not None:
            processed['image_path'] = sample['image_path']
            # Create prompt with images
            processed['prompt'] = [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(sample['image_path']))),
                    {'type': 'text', 'text': self.question_prompt.format(Question=sample['problem'])}
                ]
            }]
        else:
            # Text-only prompt
            processed['prompt'] = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': self.question_prompt.format(Question=sample['problem'])}
                ]
            }]
        
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
                # Pass epoch info to generator if it accepts it
                try:
                    # Try calling with epoch info
                    sample = self.data_generator_func(epoch=self.current_epoch, sample_idx=i)
                except TypeError:
                    # Fall back to no arguments
                    sample = self.data_generator_func()
                
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
            logger.info(f"CyclicDynamicDataset: Generating new samples for epoch {self.current_epoch}")
            
            # Generate fresh samples from base dataset
            base_iterator = iter(self.base_dataset)
            samples = []
            
            # Calculate how many unique samples we need
            effective_batch_size = len(self.base_dataset) // self.num_generations
                
            successful_samples = 0
            attempts = 0
            max_attempts = effective_batch_size * 2  # Allow some failures
            
            while successful_samples < effective_batch_size and attempts < max_attempts:
                try:
                    sample = next(base_iterator)
                    samples.append(sample)
                    successful_samples += 1
                    attempts += 1
                except StopIteration:
                    logger.warning(f"Base dataset exhausted after {successful_samples} successful samples (needed {effective_batch_size})")
                    break
                except Exception as e:
                    logger.warning(f"Failed to get sample {attempts}: {e}")
                    attempts += 1
                    continue
            
            if successful_samples == 0:
                logger.error("No samples could be generated! This will cause training to fail.")
                # Create a minimal fallback sample to prevent complete failure
                fallback_sample = {
                    'accu_reward_method': 'clevr_spotdiff',
                    'game_data': {'game_id': 'fallback', 'spy_player': 1, 'num_players': 4},
                    'metadata': {'fallback': True}
                }
                samples = [fallback_sample]
                successful_samples = 1
            
            logger.info(f"CyclicDynamicDataset: Generated {successful_samples} samples successfully")
            
            # Create the cyclic pattern
            self._cached_samples = []
            for sample in samples:
                # For each sample, repeat it num_generations * num_iterations times
                for _ in range(self.num_iterations):
                    for _ in range(self.num_generations):
                        self._cached_samples.append(sample)
            
            self._cache_epoch = self.current_epoch
            logger.info(f"CyclicDynamicDataset: Cached {len(self._cached_samples)} samples")
    
    def __iter__(self):
        """Return cached samples in cyclic pattern"""
        self._generate_cycle_samples()
        
        for sample in self._cached_samples:
            yield sample
    
    def __len__(self):
        """Return total length including repetitions"""
        base_length = len(self.base_dataset)
        effective_batch_size = base_length // self.num_generations
        return effective_batch_size * self.num_generations * self.num_iterations 