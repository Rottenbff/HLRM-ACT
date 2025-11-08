# arc_dataset.py

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from glob import glob

# Import dihedral transforms from our common module
from common import dihedral_transform, DIHEDRAL_INVERSE

# Constants
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

class ARCPuzzle:
    def __init__(self, id: str, examples: List[Tuple[np.ndarray, np.ndarray]]):
        self.id = id
        self.examples = examples

def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    """Convert ARC grid to numpy array with validation"""
    arr = np.array(grid)
    
    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)

def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    """Convert grid to sequence format with optional translation augmentation
    
    Args:
        inp: input grid
        out: output grid  
        do_translation: whether to apply translation augmentation
        
    Returns:
        tuple of (input_seq, output_seq) where seq is flattened with PAD/EOS tokens
    """
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        # Shift digits by 2 to make room for PAD(0) and EOS(1)
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), 
                                 (pad_c, ARCMaxGridSize - pad_c - ncol)), 
                     constant_values=0)

        # Add <eos> token
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result

def puzzle_hash(puzzle: ARCPuzzle):
    """Hash the puzzle for checking equivalence"""
    def _grid_hash(grid: np.ndarray):
        buffer = [x.to_bytes(1) for x in grid.shape]
        buffer.append(grid.tobytes())
        return hashlib.sha256(b"".join(buffer)).hexdigest()
    
    hashes = []
    for input, label in puzzle.examples:
        hashes.append(f"{_grid_hash(input)}|{_grid_hash(label)}")
        
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()

def convert_single_arc_puzzle(results: dict, default_name: str, puzzle: dict, 
                            aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):
    """Convert a single ARC puzzle with augmentation"""
    # Remove "name" 
    name = puzzle.pop("name", default_name)
    
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([
            (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])) 
            for example in examples
        ])

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            # Augmentation plan
            trans_id = np.random.randint(0, 8)
            # Permute colors, excluding "0" (black)
            mapping = np.concatenate([
                np.arange(0, 1, dtype=np.uint8), 
                np.random.permutation(np.arange(1, 10, dtype=np.uint8))
            ])
            
            aug_repr = f"t{trans_id}_{''.join(str(x) for x in mapping)}"

            def _map_grid(grid: np.ndarray):
                return dihedral_transform(mapping[grid], trans_id)
            
            # Check duplicate
            augmented = {}
            for dest, puzzle in converted.items():
                examples = [(_map_grid(input), _map_grid(label)) for (input, label) in puzzle.examples]
                augmented[dest] = ARCPuzzle(f"{puzzle.id}_{aug_repr}", examples)
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
                
        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append to results
    for dest in dests:
        dest_split, dest_set = dest
        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted for converted in group])

def load_puzzles_arcagi(results: dict, dataset_path: str, num_aug: int):
    """Load ARC-AGI puzzles from directory structure"""
    train_examples_dest = ("train", "all")
    test_examples_map = {
        "evaluation": [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    total_puzzles = 0
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return
    
    for subdir in os.scandir(dataset_path):
        if subdir.is_dir():
            # Load all puzzles in this directory
            puzzles = []
            for filename in glob(os.path.join(subdir.path, "*.json")):
                with open(filename, "r") as f:
                    puzzles.append((Path(filename).stem, json.load(f)))
                    
            # Shuffle puzzles
            np.random.shuffle(puzzles)
            
            # Assign by fraction
            for idx, (default_name, puzzle) in enumerate(puzzles):
                fraction = idx / len(puzzles)
                test_examples_dest = None
                for f, dest in test_examples_map.get(subdir.name, test_examples_map["_default"]):
                    if fraction < f:
                        test_examples_dest = dest
                        break
                        
                assert test_examples_dest is not None
                
                convert_single_arc_puzzle(results, default_name, puzzle, num_aug, 
                                        {"train": train_examples_dest, "test": test_examples_dest})
                total_puzzles += 1

    print(f"[{dataset_path}] total puzzles: {total_puzzles}")

def load_arc_dataset(dataset_dir: str = "dataset/raw-data/ARC-AGI", 
                    num_aug: int = 1000, 
                    seed: int = 42,
                    split: str = "train") -> List[Dict]:
    """Load ARC-AGI dataset with augmentation
    
    Args:
        dataset_dir: Directory containing ARC-AGI data
        num_aug: Number of augmentations (0 for test, 1000+ for train)
        seed: Random seed
        split: "train" or "test"
        
    Returns:
        List of examples with 'puzzle', 'solution' fields
    """
    np.random.seed(seed)
    
    # Read dataset
    data = {}
    
    # For train split, use the main training data
    # For test split, use evaluation data
    if split == "train":
        dataset_paths = [dataset_dir]
    else:  # test
        dataset_paths = [dataset_dir]
    
    for dataset_path in dataset_paths:
        load_puzzles_arcagi(data, dataset_path, num_aug)
    
    if not data:
        print(f"Warning: No data found in {dataset_dir}")
        return []
    
    # Map global puzzle identifiers
    num_identifiers = 1  # 0 is blank
    identifier_map = {}
    for split_name, split_data in data.items():
        for subset_name, subset in split_data.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1

    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Process the target split
    if split not in data:
        print(f"Warning: Split '{split}' not found in data")
        return []
    
    split_data = data[split]
    
    # Handle multiple subsets
    results = []
    for subset_name, subset in split_data.items():
        # Use first subset or specific subset
        if split == "test" and subset_name != "all":
            continue
            
        for group in subset:
            for puzzle in group:
                # Get one example per puzzle (no augmentation index 0)
                no_aug_id = np.random.randint(0, len(puzzle.examples))
                for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                    # Use first example for consistency
                    if _idx_ex == 0:
                        inp_seq, out_seq = np_grid_to_seq_translational_augment(
                            inp, out, do_translation=(split == "train" and _idx_ex != no_aug_id)
                        )
                        
                        results.append({
                            "puzzle": inp_seq,
                            "solution": out_seq,
                            "puzzle_id": identifier_map[puzzle.id]
                        })
                        break
    
    print(f"Loaded {len(results)} examples from {split} split")
    return results

def create_sample_arc_data(num_examples: int = 100) -> List[Dict]:
    """Create sample ARC data for testing when real data is not available"""
    results = []
    
    # Create simple patterns for testing
    for i in range(num_examples):
        # Create a simple 3x3 pattern
        size = 3
        input_grid = np.random.randint(1, 5, (size, size), dtype=np.uint8)
        
        # Simple transformation rule: multiply by 2
        output_grid = (input_grid * 2) % 10
        
        # Pad to sequence format
        inp_seq, out_seq = np_grid_to_seq_translational_augment(
            input_grid, output_grid, do_translation=False
        )
        
        results.append({
            "puzzle": inp_seq,
            "solution": out_seq,
            "puzzle_id": i + 1
        })
    
    return results

# For compatibility with existing code
def load_arc_agi_dataset(split: str = "train", use_augmentation: bool = True) -> List[Dict]:
    """Load ARC-AGI dataset with simplified interface"""
    num_aug = 1000 if use_augmentation and split == "train" else 0
    return load_arc_dataset(num_aug=num_aug, split=split)