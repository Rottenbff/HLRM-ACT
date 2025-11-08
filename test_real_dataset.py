#!/usr/bin/env python3
"""
Test EchoMind checkpoints on the actual training dataset (sapientinc/sudoku-extreme).
This tests on the REAL distribution the model was trained on.
"""

import argparse
import numpy as np
from sudoku import load_sapient_sudoku_dataset, print_sudoku_board
from model import HRMACTInner
from config import HRMConfig
from torch import nn
import torch

def test_checkpoint_on_dataset(
    checkpoint_path: str,
    num_samples: int = 100,
    num_augment: int = 0,  # 0 for test (no augmentation)
    min_difficulty: int = None,
    device: str = "auto"
):
    """Test a checkpoint on the actual training dataset"""

    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("=" * 80)
    print(f"Testing Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Test Samples: {num_samples}")
    print(f"Augmentations: {num_augment}")
    if min_difficulty:
        print(f"Min Difficulty: {min_difficulty}")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    ds = load_sapient_sudoku_dataset(
        subset="test",
        num_augment=num_augment,
        subsample_size=num_samples,
        min_difficulty=min_difficulty,
        batch_size=100,
        num_proc=4
    )
    print(f"Loaded {len(ds)} examples")

    # Convert to list for easier indexing
    examples = list(ds)
    print(f"Converted to {len(examples)} examples")

    # Initialize model
    print("\n[2/4] Initializing model...")
    config = HRMConfig()
    model = HRMACTInner(config)
    model.to(device)
    model.eval()

    # Load checkpoint
    print("\n[3/4] Loading checkpoint...")
    from safetensors import safe_open
    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        model.load_state_dict(f.get_tensor("model"))

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test on samples
    print("\n[4/4] Testing...")
    wins = 0
    losses = 0

    for idx, example in enumerate(examples):
        puzzle = np.array(example['puzzle'])
        solution = np.array(example['solution'])

        # Run inference
        puzzle_tensor = torch.tensor(puzzle.flatten(), dtype=torch.long, device=device).unsqueeze(0)

        # Initialize hidden states
        low_h = model.initial_low_level.unsqueeze(0).expand(1, config.seq_len + 1, -1)
        high_h = model.initial_high_level.unsqueeze(0).expand(1, config.seq_len + 1, -1)
        hidden_states = (low_h, high_h)

        with torch.no_grad():
            for segment in range(1, config.act.halt_max_steps + 1):
                output = model(hidden_states, puzzle_tensor)
                hidden_states, output_logits, q_halt, q_continue = output

                predictions = output_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                # Build predicted board
                predicted_board_flat = []
                for i, p_val in enumerate(puzzle.flatten()):
                    if p_val != 0:
                        predicted_board_flat.append(p_val)
                    else:
                        predicted_board_flat.append(predictions[i])

                predicted_board = np.array(predicted_board_flat).reshape(9, 9)

                # Check if correct
                is_correct = np.allclose(predicted_board, solution)
                q_h_val = torch.sigmoid(q_halt).item()
                q_c_val = torch.sigmoid(q_continue).item()

                if q_h_val > q_c_val:
                    break

        # Count win/loss
        if is_correct:
            wins += 1
            status = "✓"
        else:
            losses += 1
            status = "✗"

        # Print progress
        if (idx + 1) % 10 == 0 or idx == len(examples) - 1:
            print(f"  [{idx+1}/{len(examples)}] {status} {wins}/{len(examples)} ({100*wins/max(1,idx+1):.1f}%) "
                  f"Segments: {segment} Q[h,c]: {q_h_val:.3f},{q_c_val:.3f}")

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total Samples: {len(examples)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {100*wins/len(examples):.2f}%")
    print("=" * 80)

    return wins / len(examples)


def main():
    parser = argparse.ArgumentParser(description="Test EchoMind on real dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--difficulty", type=int, help="Minimum difficulty rating")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    win_rate = test_checkpoint_on_dataset(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        min_difficulty=args.difficulty,
        device=args.device
    )

    print(f"\nWin Rate: {100*win_rate:.2f}%")


if __name__ == "__main__":
    main()
