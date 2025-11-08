# train.py

import torch
import torch.nn.functional as F
import numpy as np
from model import HRMACTInner
from sudoku import load_online_puzzle, generate_sudoku, Difficulty, load_sapient_sudoku_dataset
from arc_dataset import load_arc_agi_dataset, create_sample_arc_data
import random
import multiprocessing

def arc_agi_loss(model, hidden_states, inputs, targets, segments):
    """Loss function for ARC-AGI tasks"""
    config = model.config

    (next_hidden_states, output_logits,
     q_act_halt_logits, q_act_continue_logits) = model(hidden_states, inputs)

    # Output loss for ARC-AGI prediction
    output_loss = F.cross_entropy(
        output_logits.view(-1, config.vocab_size),
        targets.view(-1),
        reduction='none'
    ).view(inputs.shape)

    # For ARC-AGI, all positions are valid (no masking like Sudoku)
    # Just take the mean over all positions
    masked_output_loss = output_loss.mean()

    # Accuracy for halting decision target - all positions must be correct
    with torch.no_grad():
        predictions = output_logits.argmax(dim=2)
        # For ARC-AGI, all predictions must match targets
        output_accuracy = (predictions == targets).all(dim=1).long()

    # Q-ACT loss
    next_segments = segments + 1
    is_last_segment = next_segments >= config.act.halt_max_steps

    is_halted = is_last_segment | (q_act_halt_logits > q_act_continue_logits)

    # Halt exploration logic
    halt_exploration = torch.rand_like(q_act_halt_logits) < config.act.halt_exploration_probability
    if halt_exploration.any():
        min_halt_segments = torch.randint(2, config.act.halt_max_steps + 1, segments.shape, device=segments.device)
        min_halt_segments = min_halt_segments.long() * halt_exploration.long()
        is_halted = is_halted & (next_segments > min_halt_segments)

    # Compute target for Q-Continue
    with torch.no_grad():
        (_, _, next_q_act_halt, next_q_act_continue) = model(next_hidden_states, inputs)

    q_act_continue_target = torch.where(
        is_last_segment,
        next_q_act_halt,
        torch.maximum(next_q_act_halt, next_q_act_continue)
    ).sigmoid()

    q_act_loss = (
        F.binary_cross_entropy_with_logits(q_act_halt_logits, output_accuracy.float(), reduction='none') + \
        F.binary_cross_entropy_with_logits(q_act_continue_logits, q_act_continue_target, reduction='none')
    ) / 2
    avg_q_act_loss = q_act_loss.mean()

    total_loss = masked_output_loss + avg_q_act_loss

    # Metrics for logging
    with torch.no_grad():
        full_accuracy = (predictions == targets).float().mean()
        q_act_halt_accuracy = ((q_act_halt_logits >= 0) == output_accuracy.bool()).float().mean()

    return (total_loss, masked_output_loss, avg_q_act_loss, is_halted,
            full_accuracy, q_act_halt_accuracy, next_hidden_states)

def sudoku_loss(model, hidden_states, board_inputs, board_targets, segments):
    config = model.config

    (next_hidden_states, output_logits,
     q_act_halt_logits, q_act_continue_logits) = model(hidden_states, board_inputs)

    # Output loss for Sudoku prediction
    output_loss = F.cross_entropy(
        output_logits.view(-1, config.vocab_size),
        board_targets.view(-1),
        reduction='none'
    ).view(board_inputs.shape)

    output_loss_mask = (board_inputs == 0).float()
    masked_output_loss = (output_loss * output_loss_mask).sum() / output_loss_mask.sum().clamp(min=1)

    # Accuracy for halting decision target
    with torch.no_grad():
        predictions = output_logits.argmax(dim=2)
        output_accuracy = ((predictions == board_targets) | (board_inputs != 0)).all(dim=1).long()

    # Q-ACT loss
    next_segments = segments + 1
    is_last_segment = next_segments >= config.act.halt_max_steps

    is_halted = is_last_segment | (q_act_halt_logits > q_act_continue_logits)

    # Halt exploration logic
    halt_exploration = torch.rand_like(q_act_halt_logits) < config.act.halt_exploration_probability
    if halt_exploration.any():
        min_halt_segments = torch.randint(2, config.act.halt_max_steps + 1, segments.shape, device=segments.device)
        min_halt_segments = min_halt_segments.long() * halt_exploration.long()
        is_halted = is_halted & (next_segments > min_halt_segments)

    # Compute target for Q-Continue
    with torch.no_grad():
        (_, _, next_q_act_halt, next_q_act_continue) = model(next_hidden_states, board_inputs)

    q_act_continue_target = torch.where(
        is_last_segment,
        next_q_act_halt,
        torch.maximum(next_q_act_halt, next_q_act_continue)
    ).sigmoid()

    q_act_loss = (
        F.binary_cross_entropy_with_logits(q_act_halt_logits, output_accuracy.float(), reduction='none') + \
        F.binary_cross_entropy_with_logits(q_act_continue_logits, q_act_continue_target, reduction='none')
    ) / 2
    avg_q_act_loss = q_act_loss.mean()

    total_loss = masked_output_loss + avg_q_act_loss

    # Metrics for logging
    with torch.no_grad():
        full_accuracy = ((predictions == board_targets) | (board_inputs != 0)).float().mean()
        q_act_halt_accuracy = ((q_act_halt_logits >= 0) == output_accuracy.bool()).float().mean()

    return (total_loss, masked_output_loss, avg_q_act_loss, is_halted,
            full_accuracy, q_act_halt_accuracy, next_hidden_states)

class TrainingBatch:
    DIFFICULTIES = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXTREME]
    CURRICULUM_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.3, 0.4, 0.2],
    ]

    def _select_puzzle_pool(self, key):
        total_cores = multiprocessing.cpu_count()
        num_proc = max(1,  total_cores - 1)
        return self.dataset.filter(lambda example: example['missing'] == key, num_proc=num_proc)

    def __init__(self, model: HRMACTInner, batch_size: int, device: torch.device, shard: str = None, use_sapient: bool = False, use_arc_agi: bool = False, subsample_size: int = None):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.curriculum_level = 0
        self.total_puzzles = 0
        self.use_arc_agi = use_arc_agi

        if use_arc_agi:
            print("Using ARC-AGI dataset")
            # Try to load real ARC-AGI data, fall back to sample data
            try:
                self.dataset = load_arc_agi_dataset(split="train", use_augmentation=True)
                if not self.dataset:
                    print("No real ARC-AGI data found, using sample data")
                    self.dataset = create_sample_arc_data(num_examples=1000)
            except Exception as e:
                print(f"Error loading ARC-AGI data: {e}, using sample data")
                self.dataset = create_sample_arc_data(num_examples=1000)
            
            self.levels = ["all"]  # Single level for ARC-AGI
            self.puzzle_pool = self.dataset
            self.sample_puzzle = self._sample_puzzle_from_dataset
        elif use_sapient:
            print("Using sapientinc/sudoku-extreme dataset with augmentation")
            self.dataset = load_sapient_sudoku_dataset(
                subset="train",
                num_augment=1000,
                subsample_size=subsample_size,
                min_difficulty=1
            )
            self.levels = ["extreme"]  # All puzzles are extreme difficulty
            self.puzzle_pool = self.dataset
            self.sample_puzzle = self._sample_puzzle_from_dataset
        elif shard is None:
            print("Sample puzzle from algorithm")
            self.sample_puzzle = self._sample_puzzle_from_algorithm
        else:
            print("Sample puzzle from dataset")
            self.dataset = load_online_puzzle(shard) if shard is not None else []
            self.levels = list(set(self.dataset['missing']))
            # self.puzzle_pool = self._select_puzzle_pool(self.levels[self.curriculum_level])
            self.puzzle_pool = self.dataset
            self.sample_puzzle = self._sample_puzzle_from_dataset

        hidden_size = model.config.transformers.hidden_size
        seq_len = model.config.max_seq_len

        # For ARC-AGI, use variable length support
        if use_arc_agi:
            # Find the maximum sequence length in the dataset
            max_len = 0
            for item in self.puzzle_pool:
                max_len = max(max_len, len(item['puzzle']))
            self.max_seq_len = min(max_len, model.config.target_seq_len)  # Use model's target seq len
            
            # Model expects max_seq_len + 1 for class token
            self.model_max_seq_len = model.config.max_seq_len
            
            self.board_inputs = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long, device=device)
            self.board_targets = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long, device=device)
            
            # Initialize hidden states with model max_seq_len (includes class token)
            low_level_h = torch.zeros((batch_size, self.model_max_seq_len, hidden_size), dtype=model.config.dtype, device=device)
            high_level_h = torch.zeros((batch_size, self.model_max_seq_len, hidden_size), dtype=model.config.dtype, device=device)
            self.hidden_states = (low_level_h, high_level_h)
        else:
            # Sudoku configuration
            self.max_seq_len = seq_len
            self.board_inputs = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            self.board_targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            
            # FIX 1: Initialize hidden_states with new, independent tensors, not views of model parameters.
            low_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
            high_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
            self.hidden_states = (low_level_h, high_level_h)
        
        self.segments = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            self.replace(i)

    def _sample_difficulty(self):
        return random.choices(self.DIFFICULTIES, self.CURRICULUM_PROBAS[self.curriculum_level], k=1)[0]

    def _normalize_dataset_to_np_array(self, sample):
        return np.array(list(map(int, sample)), dtype=np.int64).reshape(9, 9),

    def _sample_puzzle_from_dataset(self, idx):
        actual_idx = (self.total_puzzles + idx) % len(self.puzzle_pool)
        print(f"Sampling puzzle from dataset at index {actual_idx}")
        if self.use_arc_agi:
            # For ARC-AGI, the data is already in sequence format
            return (
                self.puzzle_pool[actual_idx]['puzzle'],
                self.puzzle_pool[actual_idx]['solution']
            )
        else:
            return (
                np.array(self.puzzle_pool[actual_idx]['puzzle']),
                np.array(self.puzzle_pool[actual_idx]['solution'])
            )

    def _sample_puzzle_from_algorithm(self, difficulty, idx):
        return generate_sudoku(self._sample_difficulty())

    def replace(self, idx: int):
        puzzle, solution = self.sample_puzzle(idx)
        
        if self.use_arc_agi:
            # For ARC-AGI, pad to max_seq_len
            puzzle = np.array(puzzle)
            solution = np.array(solution)
            
            # Ensure sequences are not longer than max_seq_len
            puzzle = puzzle[:self.max_seq_len]
            solution = solution[:self.max_seq_len]
            
            # Pad to max_seq_len
            puzzle = np.pad(puzzle, (0, self.max_seq_len - len(puzzle)), constant_values=0)
            solution = np.pad(solution, (0, self.max_seq_len - len(solution)), constant_values=0)
        
        self.board_inputs[idx] = torch.tensor(puzzle.flatten(), device=self.device)
        self.board_targets[idx] = torch.tensor(solution.flatten(), device=self.device)

        # FIX 2: Wrap the state reset in a `torch.no_grad()` context.
        with torch.no_grad():
            self.segments[idx] = 0

            # Use model max_seq_len for hidden states
            seq_len = self.model_max_seq_len
            low_level_h, high_level_h = self.hidden_states

            # This assignment correctly resets the state for one sample in the batch
            # using the model's learnable initial state parameters.
            low_level_h[idx] = self.model.initial_low_level.unsqueeze(0).expand(seq_len, -1)
            high_level_h[idx] = self.model.initial_high_level.unsqueeze(0).expand(seq_len, -1)

        self.total_puzzles += 1

    def graduate(self):
        if self.puzzle_pool is not None and self.curriculum_level + 1 < len(self.levels):
            self.curriculum_level += 1
            print(f"Graduated to curriculum level {self.curriculum_level}.")
        elif self.curriculum_level + 1 < len(self.CURRICULUM_PROBAS):
            self.curriculum_level += 1
            print(f"Graduated to curriculum level {self.curriculum_level}.")
        else:
            print("Reached highest curriculum level.")

def train_step(model, optimizer, batch):
    optimizer.zero_grad()

    if batch.use_arc_agi:
        loss_fn = arc_agi_loss
    else:
        loss_fn = sudoku_loss

    (loss, out_loss, q_loss, is_halted,
     out_acc, q_acc, next_h) = loss_fn(
        model,
        batch.hidden_states,
        batch.board_inputs,
        batch.board_targets,
        batch.segments
    )

    loss.backward()
    optimizer.step()

    print(
        f"Output [Loss: {out_loss.item():.4f}, Acc: {out_acc.item():.4f}] | "
        f"Q-ACT [Loss: {q_loss.item():.4f}, Acc: {q_acc.item():.4f}] | "
        f"Puzzles [{batch.total_puzzles}] | Curriculum [{batch.curriculum_level}]"
    )

    batch.hidden_states = next_h
    batch.segments += 1

    halted_indices = torch.where(is_halted)[0]
    for idx in halted_indices:
        batch.replace(idx.item())

    return out_acc.item()