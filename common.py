# common.py

from typing import List, Optional
import pydantic
import numpy as np

# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    
    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr

def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply the inverse of a dihedral transform"""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

def apply_color_permutation(grid: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Apply a color permutation to a grid"""
    return mapping[grid]

def create_random_color_mapping(rng: np.random.Generator) -> np.ndarray:
    """Create a random color mapping (permutation of 1-9, keeping 0 as is)"""
    return np.concatenate([
        np.array([0], dtype=np.uint8),  # Keep background (0) unchanged
        rng.permutation(np.arange(1, 10, dtype=np.uint8))  # Permute colors 1-9
    ])

def generate_random_transformation(rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Generate random dihedral transformation and color mapping"""
    trans_id = rng.integers(0, 8)  # 0-7 for dihedral transforms
    color_mapping = create_random_color_mapping(rng)
    return trans_id, color_mapping

def apply_augmentation(grid: np.ndarray, trans_id: int, color_mapping: np.ndarray) -> np.ndarray:
    """Apply dihedral transformation and color permutation to a grid"""
    # Apply color permutation first
    if color_mapping is not None:
        grid = apply_color_permutation(grid, color_mapping)
    
    # Apply dihedral transformation
    if trans_id != 0:
        grid = dihedral_transform(grid, trans_id)
    
    return grid

def create_variable_size_padding(grid: np.ndarray, max_size: int = 30) -> np.ndarray:
    """Pad grid to square with size <= max_size"""
    if grid.shape[0] > max_size or grid.shape[1] > max_size:
        raise ValueError(f"Grid size {grid.shape} exceeds maximum size {max_size}")
    
    # Make square by padding
    size = max(grid.shape)
    padded = np.zeros((size, size), dtype=grid.dtype)
    pad_h = (size - grid.shape[0]) // 2
    pad_w = (size - grid.shape[1]) // 2
    padded[pad_h:pad_h + grid.shape[0], pad_w:pad_w + grid.shape[1]] = grid
    return padded

def grid_to_sequence(grid: np.ndarray, max_size: int = 30) -> np.ndarray:
    """Convert grid to sequence format with padding and EOS token"""
    # Pad to square
    padded_grid = create_variable_size_padding(grid, max_size)
    
    # Add EOS token at the end (use a special value, typically 1)
    eos_token = 1
    size = padded_grid.shape[0]
    
    # Create sequence: grid values + EOS
    sequence = padded_grid.flatten()
    
    # Find first zero (background) to place EOS
    # If no zeros, place at end
    eos_pos = size * size  # Default to end
    for i, val in enumerate(sequence):
        if val == 0:
            eos_pos = i
            break
    
    # Insert EOS token
    sequence_with_eos = np.insert(sequence, eos_pos, eos_token)
    
    return sequence_with_eos

def sequence_to_grid(sequence: np.ndarray, max_size: int = 30) -> np.ndarray:
    """Convert sequence back to grid format"""
    eos_token = 1
    
    # Find EOS token position
    eos_pos = np.where(sequence == eos_token)[0]
    if len(eos_pos) > 0:
        eos_pos = eos_pos[0]
        sequence = sequence[:eos_pos]
    else:
        # No EOS found, use sequence as is
        pass
    
    # Determine grid size
    size = int(np.sqrt(len(sequence)))
    if size * size != len(sequence):
        # Not a perfect square, find the largest possible grid
        size = int(np.sqrt(max_size * max_size))
        while size * size > len(sequence) and size > 0:
            size -= 1
    
    return sequence[:size*size].reshape(size, size)

def arc_accuracy_metric(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy for ARC tasks (exact match)"""
    if predictions.shape != targets.shape:
        return 0.0
    
    # Count exact matches
    exact_matches = np.sum(predictions == targets)
    total_elements = predictions.size
    
    return exact_matches / total_elements if total_elements > 0 else 0.0

def create_arc_batch_collate_fn(max_seq_len: int = 900):
    """Create a collate function for ARC batches with variable sequence lengths"""
    def collate_fn(batch):
        # Extract sequences and targets
        inputs = []
        targets = []
        lengths = []
        
        for item in batch:
            inputs.append(item['puzzle'])
            targets.append(item['solution'])
            lengths.append(len(item['puzzle']))
        
        # Pad sequences to the same length
        max_len = min(max(lengths), max_seq_len)
        
        padded_inputs = np.full((len(batch), max_len), 0, dtype=np.int64)
        padded_targets = np.full((len(batch), max_len), 0, dtype=np.int64)
        
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            actual_len = min(len(inp), max_seq_len)
            padded_inputs[i, :actual_len] = inp[:actual_len]
            padded_targets[i, :actual_len] = tgt[:actual_len]
        
        return {
            'inputs': padded_inputs,
            'targets': padded_targets,
            'lengths': np.array(lengths)
        }
    
    return collate_fn