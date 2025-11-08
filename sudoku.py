# sudoku.py

import random
import numpy as np
from enum import Enum
from datasets import load_dataset
import multiprocessing

class Difficulty(Enum):
    VERY_EASY = (46, 50)
    EASY = (40, 45)
    MEDIUM = (32, 39)
    HARD = (28, 31)
    EXTREME = (17, 27)

def _get_masks(grid):
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
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                return r, c
    return None

def _fill_grid_recursive(grid, rows, cols, boxes):
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

def generate_sudoku(difficulty: Difficulty):
    board = np.zeros((9, 9), dtype=int)
    rows, cols, boxes = _get_masks(board)
    _fill_grid_recursive(board, rows, cols, boxes)

    solution = board.copy()
    puzzle = board.copy()

    target_clues_range = difficulty.value

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

def load_online_puzzle(shard: str, batch_size: int = 1_000, num_proc: int = None) -> dict:
    """
    Load a Sudoku dataset from an online source and group puzzles by their 'missing' count.

    Returns:
        grouped_data: dict where keys are missing counts (as strings)
                      and values are lists of samples (dicts).
                      Each sample includes:
                        - 'puzzle': np.ndarray (9, 9)
                        - 'solution': np.ndarray (9, 9)
                        - 'missing', 'difficulty', 'set', 'solving_time'
    """
    total_cores = multiprocessing.cpu_count()
    num_proc = max(1, (num_proc or total_cores - 1))

    # Load the dataset from Hugging Face
    print("[INFO] Loading dataset...")
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split=shard)
    print("[INFO] Dataset remove columns.")
    ds = ds.select_columns(['puzzle', 'solution', 'missing'])
    def convert_batch(batch):
        puzzles, solutions = [], []
        for p_str, s_str in zip(batch["puzzle"], batch["solution"]):
            puzzles.append(np.fromiter(map(int, p_str), dtype=np.int64).reshape(9, 9))
            solutions.append(np.fromiter(map(int, s_str), dtype=np.int64).reshape(9, 9))
        return {"puzzle": puzzles, "solution": solutions}

    # Convert 'puzzle' and 'solution' to 9x9 numpy arrays
    print("[INFO] Dataset format conversion.")
    ds = ds.map(convert_batch, batched=True, batch_size=batch_size, num_proc=num_proc)
    print("[INFO] Dataset set remove coulumns is too easy")
    ds = ds.filter(lambda example: example['missing'] != 1, num_proc=num_proc)
    return ds


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, rng=None):
    """
    Apply Sudoku augmentation transformations.

    Transformations:
    - Random digit mapping (permutation of 1..9)
    - Random transposition
    - Row band shuffling (3 bands × 3 rows each)
    - Column stack shuffling (3 stacks × 3 columns each)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose
    transpose_flag = rng.random() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])

    # Similarly for columns (stacks)
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])

    # Build an 81->81 mapping
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def load_sapient_sudoku_dataset(
    subset: str = "train",
    num_augment: int = 1000,
    subsample_size: int = None,
    min_difficulty: int = None,
) -> list:
    """
    Load Sudoku dataset from sapientinc/sudoku-extreme with augmentation.
    This replicates the exact logic from HRM's build_sudoku_dataset.py

    Args:
        subset: 'train' or 'test'
        num_augment: Number of augmentations (0 for test, 1000 for train)
        subsample_size: Randomly sample this many examples
        min_difficulty: Minimum difficulty rating

    Returns:
        List of examples with 'puzzle' and 'solution' fields
    """
    import csv
    from huggingface_hub import hf_hub_download

    # Read CSV using exact HRM logic
    inputs = []
    labels = []

    print(f"[INFO] Loading {subset} split from sapientinc/sudoku-extreme...")
    csv_path = hf_hub_download("sapientinc/sudoku-extreme", f"{subset}.csv", repo_type="dataset")

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                assert len(q) == 81 and len(a) == 81

                # Convert puzzle (replace '.' with '0') and solution using exact HRM logic
                puzzle = np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
                solution = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
                inputs.append(puzzle)
                labels.append(solution)

    print(f"[INFO] Loaded {len(inputs)} examples")

    # If subsample_size is specified for the training set
    if subset == "train" and subsample_size is not None and subsample_size < len(inputs):
        print(f"[INFO] Subsampling to {subsample_size} examples")
        indices = np.random.choice(len(inputs), size=subsample_size, replace=False)
        inputs = [inputs[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Generate dataset with augmentations using exact HRM logic
    num_augments = num_augment if subset == "train" else 0

    results = []
    for orig_inp, orig_out in zip(inputs, labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results.append({
                "puzzle": inp,
                "solution": out
            })

    print(f"[INFO] Total examples after augmentation: {len(results)}")
    return results


def sudoku_board_string(board):
    horizontal_line = "+-------+-------+-------+"
    result = horizontal_line + "\n"
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
