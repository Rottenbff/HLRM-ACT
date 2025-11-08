#!/usr/bin/env python3
"""
ECHOMIND INFERENCE ENGINE
========================
Single-file, portable inference script for the Hierarchical Reasoning Model.
No external dependencies beyond PyTorch, safetensors, and adam_atan2_pytorch.

Usage:
    python inference_engine.py --checkpoint model.safetensors --difficulty easy --turns 1 --verbose

Features:
- Standalone Sudoku puzzle generation (5 difficulty levels)
- Pre-defined sample puzzles for testing (use --use-samples flag)
- Model inference with ACT (Adaptive Computation Time)
- Real-time Q-value monitoring
- Win/loss tracking
- No FlashAttention required

Sample Puzzles (use --use-samples):
- very-easy: Classic example (37 clues)
- medium: Medium challenge (30 clues)
- hard: Hard puzzle (24 clues)
- extreme: Extreme challenge (17 clues!)

Examples:
    # Use a sample puzzle
    python inference_engine.py --checkpoint model.safetensors --difficulty hard --use-samples --verbose

    # Generate and solve a random puzzle
    python inference_engine.py --checkpoint model.safetensors --difficulty easy
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import random
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HRMACTModelConfig:
    @dataclass
    class TransformerConfig:
        num_layers: int
        hidden_size: int
        num_heads: int
        expansion: float = 4.0
        norm_epsilon: float = 1e-5
        rope_theta: float = 10000.0

    @dataclass
    class ACTConfig:
        halt_max_steps: int
        halt_exploration_probability: float

    seq_len: int
    vocab_size: int
    high_level_cycles: int
    low_level_cycles: int
    transformers: TransformerConfig
    act: ACTConfig
    dtype: torch.dtype = torch.bfloat16


# ============================================================================
# CORE MODEL ARCHITECTURE
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.epsilon = eps

    def forward(self, x):
        original_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.epsilon)).to(original_dtype)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for attention mechanisms"""
    def __init__(self, dim: int, max_length: int, base: float, dtype: torch.dtype):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
        t = torch.arange(max_length, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(-2)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        return (x * self.cos) + (self._rotate_half(x) * self.sin)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE"""
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, (num_heads * 2 + num_heads) * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x, rope):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, L, self.num_heads * 3, self.head_dim)

        query, key, value = qkv.split([self.num_heads, self.num_heads, self.num_heads], dim=2)

        query = rope(query)
        key = rope(key)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value)

        output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, expansion: float):
        super().__init__()
        hidden_dim = int(expansion * dim * 2.0 / 3.0)
        hidden_dim = (-(hidden_dim // -256)) * 256

        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class HRMACTBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config: HRMACTModelConfig.TransformerConfig):
        super().__init__()
        self.self_attn = Attention(
            dim=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads
        )
        self.mlp = SwiGLU(dim=config.hidden_size, expansion=config.expansion)
        self.norm1 = RMSNorm(eps=config.norm_epsilon)
        self.norm2 = RMSNorm(eps=config.norm_epsilon)

    def forward(self, x, rope):
        x = self.norm1(x + self.self_attn(x, rope))
        x = self.norm2(x + self.mlp(x))
        return x


class HRMACTReasoner(nn.Module):
    """Hierarchical reasoning module"""
    def __init__(self, config: HRMACTModelConfig.TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([HRMACTBlock(config) for _ in range(config.num_layers)])

    def forward(self, hidden_state, input_injection, rope):
        hidden_state = hidden_state + input_injection
        for layer in self.layers:
            hidden_state = layer(hidden_state, rope)
        return hidden_state


class HRMACTInner(nn.Module):
    """Complete HRM model with ACT"""
    def __init__(self, config: HRMACTModelConfig):
        super().__init__()
        self.config = config

        self.cls_token = nn.Parameter(torch.empty(config.transformers.hidden_size))

        self.input_embedding = nn.Embedding(config.vocab_size, config.transformers.hidden_size)

        self.output_head = nn.Linear(config.transformers.hidden_size, config.vocab_size, bias=False)

        self.q_act_head = nn.Linear(config.transformers.hidden_size, 2)

        self.rotary_emb = RotaryPositionEmbedding(
            dim=config.transformers.hidden_size // config.transformers.num_heads,
            max_length=config.seq_len + 1,
            base=config.transformers.rope_theta,
            dtype=config.dtype
        )

        self.high_level_reasoner = HRMACTReasoner(config.transformers)
        self.low_level_reasoner = HRMACTReasoner(config.transformers)

        self.initial_high_level = nn.Parameter(torch.empty(config.transformers.hidden_size))
        self.initial_low_level = nn.Parameter(torch.empty(config.transformers.hidden_size))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)

        if hasattr(self, 'initial_high_level'):
            torch.nn.init.trunc_normal_(self.initial_high_level, std=1.0, a=-2.0, b=2.0)
            torch.nn.init.trunc_normal_(self.initial_low_level, std=1.0, a=-2.0, b=2.0)
            if self.cls_token.numel() > 0:
                torch.nn.init.trunc_normal_(self.cls_token, std=1.0 / math.sqrt(self.config.transformers.hidden_size))

        if hasattr(self, 'q_act_head'):
            if self.q_act_head.weight.numel() > 0:
                nn.init.zeros_(self.q_act_head.weight)
                nn.init.zeros_(self.q_act_head.bias)


    def forward(self, hidden_states, inputs):
        low_level_z, high_level_z = hidden_states
        batch_size = inputs.shape[0]

        input_embeddings = self.input_embedding(inputs)

        cls_tokens = self.cls_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        full_embeddings = torch.cat([cls_tokens, input_embeddings], dim=1)
        full_embeddings *= math.sqrt(self.config.transformers.hidden_size)

        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles

        for cycle in range(1, total_cycles):
            low_level_z = self.low_level_reasoner(
                hidden_state=low_level_z,
                input_injection=high_level_z + full_embeddings,
                rope=self.rotary_emb
            )
            if cycle % self.config.low_level_cycles == 0:
                high_level_z = self.high_level_reasoner(
                    hidden_state=high_level_z,
                    input_injection=low_level_z,
                    rope=self.rotary_emb
                )

        low_level_z = low_level_z.detach()
        high_level_z = high_level_z.detach()

        low_level_z = self.low_level_reasoner(
            hidden_state=low_level_z,
            input_injection=high_level_z + full_embeddings,
            rope=self.rotary_emb
        )
        high_level_z = self.high_level_reasoner(
            hidden_state=high_level_z,
            input_injection=low_level_z,
            rope=self.rotary_emb
        )

        output_logits = self.output_head(high_level_z[:, 1:])
        q_act_logits = self.q_act_head(high_level_z[:, 0])

        q_act_halt = q_act_logits[:, 0]
        q_act_continue = q_act_logits[:, 1]

        new_hidden_states = (low_level_z.detach(), high_level_z.detach())

        return new_hidden_states, output_logits, q_act_halt, q_act_continue


# ============================================================================
# SAMPLE PUZZLES (Pre-defined test cases)
# ============================================================================

SAMPLE_PUZZLES = {
    "very-easy": {
        "puzzle": [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ],
        "solution": [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ],
        "name": "Classic Example"
    },
    "medium": {
        "puzzle": [
            [0, 0, 0, 6, 0, 0, 4, 0, 0],
            [7, 0, 0, 0, 0, 3, 6, 0, 0],
            [0, 0, 0, 0, 9, 1, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 0, 3, 0, 6, 0, 4, 5],
            [0, 4, 0, 2, 0, 0, 0, 6, 0],
            [9, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 0]
        ],
        "solution": [
            [1, 3, 8, 6, 7, 2, 4, 9, 5],
            [7, 9, 5, 8, 4, 3, 6, 2, 1],
            [2, 6, 4, 5, 9, 1, 7, 8, 3],
            [3, 7, 9, 4, 5, 8, 2, 1, 6],
            [4, 5, 6, 1, 8, 7, 9, 2, 3],
            [8, 1, 2, 3, 6, 9, 5, 4, 7],
            [5, 4, 1, 2, 3, 9, 8, 6, 2],
            [9, 8, 3, 7, 1, 4, 3, 5, 2],
            [6, 2, 7, 9, 4, 5, 1, 3, 8]
        ],
        "name": "Medium Challenge"
    },
    "hard": {
        "puzzle": [
            [0, 0, 0, 0, 0, 6, 0, 0, 0],
            [0, 5, 9, 0, 0, 0, 0, 0, 8],
            [2, 0, 0, 0, 0, 8, 0, 0, 0],
            [0, 4, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 7, 4],
            [0, 0, 0, 3, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 4, 0]
        ],
        "solution": [
            [4, 3, 7, 5, 1, 6, 8, 2, 9],
            [6, 5, 9, 7, 2, 4, 3, 1, 8],
            [2, 8, 1, 9, 3, 6, 4, 5, 7],
            [8, 4, 5, 6, 7, 2, 9, 3, 1],
            [7, 1, 3, 4, 8, 9, 6, 2, 5],
            [9, 2, 6, 1, 5, 3, 8, 7, 4],
            [1, 7, 4, 3, 9, 8, 2, 5, 6],
            [5, 6, 8, 2, 4, 7, 1, 9, 3],
            [9, 3, 2, 8, 6, 1, 5, 4, 7]
        ],
        "name": "Hard Puzzle"
    },
    "extreme": {
        "puzzle": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 7, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0]
        ],
        "solution": [
            [9, 6, 3, 4, 5, 7, 2, 1, 8],
            [4, 5, 7, 1, 8, 2, 6, 3, 9],
            [8, 2, 1, 6, 3, 9, 4, 7, 5],
            [3, 7, 4, 9, 1, 5, 8, 6, 2],
            [6, 1, 8, 2, 7, 3, 9, 5, 4],
            [2, 9, 5, 8, 4, 6, 3, 7, 1],
            [1, 3, 2, 5, 6, 8, 7, 4, 9],
            [5, 4, 6, 7, 3, 1, 9, 2, 8],
            [7, 8, 9, 3, 2, 4, 5, 1, 6]
        ],
        "name": "Extreme Challenge (Only 17 clues!)"
    }
}


# ============================================================================
# SUDOKU PUZZLE GENERATION
# ============================================================================

class Difficulty:
    VERY_EASY = (46, 50)
    EASY = (40, 45)
    MEDIUM = (32, 39)
    HARD = (28, 31)
    EXTREME = (17, 27)


def _get_masks(grid):
    """Get bitmasks for rows, columns, and boxes"""
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                val_bit = 1 << (grid[r, c] - 1)
                rows[r] |= val_bit
                cols[c] |= val_bit
                box_idx = (r // 3) * 3 + c // 3
                boxes[box_idx] |= val_bit
    return rows, cols, boxes


def _find_empty(grid):
    """Find first empty cell"""
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                return r, c
    return None


def _fill_grid_recursive(grid, rows, cols, boxes):
    """Fill grid recursively"""
    empty = _find_empty(grid)
    if not empty:
        return True
    r, c = empty

    box_idx = (r // 3) * 3 + c // 3
    used = rows[r] | cols[c] | boxes[box_idx]

    nums = list(range(1, 10))
    random.shuffle(nums)

    for num in nums:
        val_bit = 1 << (num - 1)
        if (used & val_bit) == 0:
            grid[r, c] = num
            rows[r] |= val_bit
            cols[c] |= val_bit
            boxes[box_idx] |= val_bit

            if _fill_grid_recursive(grid, rows, cols, boxes):
                return True

            grid[r, c] = 0
            rows[r] &= ~val_bit
            cols[c] &= ~val_bit
            boxes[box_idx] &= ~val_bit
    return False


def _solve_recursive(grid, solutions, limit, rows, cols, boxes):
    """Count solutions up to limit"""
    if solutions[0] >= limit:
        return True

    empty = _find_empty(grid)
    if not empty:
        solutions[0] += 1
        return solutions[0] >= limit

    r, c = empty
    box_idx = (r // 3) * 3 + c // 3
    used = rows[r] | cols[c] | boxes[box_idx]

    for num in range(1, 10):
        val_bit = 1 << (num - 1)
        if (used & val_bit) == 0:
            grid[r, c] = num
            rows[r] |= val_bit
            cols[c] |= val_bit
            boxes[box_idx] |= val_bit

            if _solve_recursive(grid, solutions, limit, rows, cols, boxes):
                return True

            grid[r, c] = 0
            rows[r] &= ~val_bit
            cols[c] &= ~val_bit
            boxes[box_idx] &= ~val_bit

    return False


def generate_sudoku(difficulty: Tuple[int, int]):
    """Generate a Sudoku puzzle with given difficulty"""
    board = np.zeros((9, 9), dtype=int)
    rows, cols, boxes = _get_masks(board)
    _fill_grid_recursive(board, rows, cols, boxes)

    solution = board.copy()
    puzzle = board.copy()

    target_clues_range = difficulty

    cells = list(range(81))
    random.shuffle(cells)

    clues = 81
    cursor = 0

    while cursor < len(cells) and clues > target_clues_range[1]:
        idx = cells[cursor]
        cursor += 1
        r, c = idx // 9, idx % 9

        backup = puzzle[r, c]
        puzzle[r, c] = 0

        test_puzzle = puzzle.copy()
        solutions = [0]

        test_rows, test_cols, test_boxes = _get_masks(test_puzzle)
        _solve_recursive(test_puzzle, solutions, 2, test_rows, test_cols, test_boxes)

        if solutions[0] != 1:
            puzzle[r, c] = backup
        else:
            clues -= 1

    if clues > target_clues_range[0]:
        for j in range(cursor, len(cells)):
            if clues <= target_clues_range[0]:
                break
            idx = cells[j]
            r, c = idx // 9, idx % 9

            if puzzle[r,c] == 0:
                continue

            backup = puzzle[r, c]
            puzzle[r, c] = 0

            test_puzzle = puzzle.copy()
            solutions = [0]

            test_rows, test_cols, test_boxes = _get_masks(test_puzzle)
            _solve_recursive(test_puzzle, solutions, 2, test_rows, test_cols, test_boxes)

            if solutions[0] != 1:
                puzzle[r, c] = backup
            else:
                clues -= 1

    return puzzle, solution


def print_sudoku_board(board, title="Board"):
    """Pretty print a Sudoku board"""
    horizontal_line = "+-------+-------+-------+"
    result = f"{title}\n{horizontal_line}\n"
    for i, row in enumerate(board):
        line = "|"
        for j, cell in enumerate(row):
            display_value = "." if cell == 0 else str(int(cell))
            line += f" {display_value}"
            if (j + 1) % 3 == 0:
                line += " |"
        result += line + "\n"
        if (i + 1) % 3 == 0:
            result += horizontal_line + "\n"
    return result.strip()


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Standalone inference engine for HRM model"""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """Initialize inference engine"""
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")

        # Model configuration
        self.config = HRMACTModelConfig(
            seq_len=81,
            vocab_size=10,
            high_level_cycles=2,
            low_level_cycles=2,
            transformers=HRMACTModelConfig.TransformerConfig(
                num_layers=4,
                hidden_size=256,
                num_heads=4,
                expansion=4
            ),
            act=HRMACTModelConfig.ACTConfig(
                halt_max_steps=16,
                halt_exploration_probability=0.1
            )
        )

        # Load model
        self.model = HRMACTInner(self.config).to(self.device, dtype=self.config.dtype)

        # Load checkpoint
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path, device=str(self.device))
        except ImportError:
            print("[WARNING] safetensors not found, trying PyTorch format...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[INFO] Loaded model from {checkpoint_path}")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def solve_puzzle(self, puzzle, solution=None, verbose: bool = False) -> Tuple[np.ndarray, bool]:
        """Solve a Sudoku puzzle"""
        if solution is None:
            solution = puzzle.copy()

        puzzle_tensor = torch.tensor(puzzle.flatten(), dtype=torch.long, device=self.device).unsqueeze(0)

        # Initial hidden states
        low_h = self.model.initial_low_level.unsqueeze(0).expand(1, self.config.seq_len + 1, -1)
        high_h = self.model.initial_high_level.unsqueeze(0).expand(1, self.config.seq_len + 1, -1)
        hidden_states = (low_h, high_h)

        with torch.no_grad():
            for segment in range(1, self.config.act.halt_max_steps + 1):
                if verbose:
                    print(f"\n--- Segment {segment} ---")

                output = self.model(hidden_states, puzzle_tensor)
                hidden_states, output_logits, q_halt, q_continue = output

                predictions = output_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                # Build predicted board
                predicted_board_flat = []
                accurate_squares, predicted_squares = 0, 0

                for i, p_val in enumerate(puzzle.flatten()):
                    if p_val != 0:
                        predicted_board_flat.append(p_val)
                    else:
                        pred = predictions[i]
                        sol = solution.flatten()[i]
                        predicted_board_flat.append(pred)
                        predicted_squares += 1
                        if pred == sol:
                            accurate_squares += 1

                predicted_board = np.array(predicted_board_flat).reshape(9, 9)

                if verbose:
                    print(f"Accuracy: {accurate_squares} / {predicted_squares} ({100*accurate_squares/max(1,predicted_squares):.1f}%)")

                q_h, q_c = torch.sigmoid(q_halt).item(), torch.sigmoid(q_continue).item()
                if verbose:
                    print(f"Q-values (halt - continue): {q_h:.4f} - {q_c:.4f}")

                if q_h > q_c:
                    if verbose:
                        print("Model halted.")
                    break

        win = np.allclose(predicted_board, solution)
        return predicted_board, win

    def generate_and_solve(self, difficulty_str: str, use_samples: bool = False, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Generate a puzzle and solve it"""
        # Try to use sample puzzle if available
        if use_samples and difficulty_str in SAMPLE_PUZZLES:
            puzzle = np.array(SAMPLE_PUZZLES[difficulty_str]["puzzle"], dtype=int)
            solution = np.array(SAMPLE_PUZZLES[difficulty_str]["solution"], dtype=int)
            name = SAMPLE_PUZZLES[difficulty_str]["name"]

            if verbose:
                print(f"\n[SAMPLE PUZZLE] {name}")
        else:
            # Generate puzzle
            difficulty_map = {
                "very-easy": Difficulty.VERY_EASY,
                "easy": Difficulty.EASY,
                "medium": Difficulty.MEDIUM,
                "hard": Difficulty.HARD,
                "extreme": Difficulty.EXTREME
            }

            difficulty = difficulty_map[difficulty_str]
            puzzle, solution = generate_sudoku(difficulty)
            name = "Generated Puzzle"

            if verbose:
                print(f"\n[GENERATED PUZZLE] Difficulty: {difficulty_str}")

        if verbose:
            print("\n" + print_sudoku_board(puzzle, "PUZZLE"))
            print("\n" + print_sudoku_board(solution, "GROUND TRUTH"))

        predicted, win = self.solve_puzzle(puzzle, solution=solution, verbose=verbose)

        if verbose:
            print("\n" + print_sudoku_board(predicted, "PREDICTION"))
            print(f"\n{'[OK] WIN' if win else '[ERROR] LOSS'}")

        return puzzle, predicted, win


def list_sample_puzzles():
    """List all available sample puzzles"""
    print("\n" + "=" * 80)
    print("AVAILABLE SAMPLE PUZZLES")
    print("=" * 80)

    for difficulty, data in SAMPLE_PUZZLES.items():
        puzzle = np.array(data["puzzle"])
        clues = np.count_nonzero(puzzle)
        name = data["name"]
        print(f"\n{difficulty.upper():12s} - {name}")
        print(f"              Clues: {clues}/81")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("To use a sample puzzle, add: --use-samples --difficulty <level>")
    print("=" * 80 + "\n")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="ECHOMIND Inference Engine - Solve Sudoku puzzles with AI"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.safetensors or .pth)")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["very-easy", "easy", "medium", "hard", "extreme"],
                        help="Puzzle difficulty level")
    parser.add_argument("--turns", type=int, default=1,
                        help="Number of puzzles to solve")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use for inference")
    parser.add_argument("--use-samples", action="store_true",
                        help="Use pre-defined sample puzzles instead of generating new ones")
    parser.add_argument("--list-samples", action="store_true",
                        help="List all available sample puzzles and exit")

    args = parser.parse_args()

    # List sample puzzles and exit if requested
    if args.list_samples:
        list_sample_puzzles()
        return

    # Validate checkpoint is provided
    if args.checkpoint is None:
        print("[ERROR] --checkpoint is required (unless using --list-samples)")
        parser.print_help()
        return

    # Initialize engine
    print("=" * 80)
    print("ECHOMIND INFERENCE ENGINE")
    print("=" * 80)
    engine = InferenceEngine(args.checkpoint, device=args.device)

    # Run inference
    wins = 0
    for turn in range(args.turns):
        print(f"\n{'=' * 80}")
        print(f"TURN {turn + 1}/{args.turns}")
        print(f"{'=' * 80}")

        _, predicted, win = engine.generate_and_solve(
            args.difficulty,
            use_samples=args.use_samples,
            verbose=args.verbose
        )
        wins += 1 if win else 0

    # Summary
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Total wins: {wins}/{args.turns} ({100*wins/max(1,args.turns):.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
