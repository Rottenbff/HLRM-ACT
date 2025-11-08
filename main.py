# main.py

import argparse
import torch
import numpy as np
from collections import deque
from safetensors.torch import save_file, load_file

from model import HRMACTInner, HRMACTModelConfig
from train import TrainingBatch, train_step
from sudoku import generate_sudoku, Difficulty, sudoku_board_string
from adam_atan2_pytorch import AdamAtan2 # adam-atan2-pytorch

def train(use_sapient=False):
    # Setup device - CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Model configuration
    config = HRMACTModelConfig(
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

    # Initialize model and move to device
    torch.manual_seed(42)
    model = HRMACTInner(config).to(device, dtype=config.dtype)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Optimizer
    optimizer = AdamAtan2(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    # Training batch
    if use_sapient:
        print("Using sapientinc/sudoku-extreme dataset")
        batch = TrainingBatch(model, batch_size=128, device=device, use_sapient=True, subsample_size=1000)
    else:
        print("Using Ritvik19/Sudoku-Dataset")
        batch = TrainingBatch(model, batch_size=128, device=device, shard="train[:1%]")

    step_idx = 0
    steps_since_graduation = 0
    accuracy_history = deque([0.0] * 300, maxlen=300)

    while True:
        step_idx += 1
        steps_since_graduation += 1
        print(f"--- Step {step_idx} ---")

        model.train()
        output_acc = train_step(model, optimizer, batch)

        if step_idx == 1 or step_idx % 250 == 0:
            print(f"Saving checkpoint at step {step_idx}...")
            save_file(model.state_dict(), f"checkpoint-{step_idx}.safetensors")

        accuracy_history.append(output_acc)
        avg_rolling_accuracy = sum(accuracy_history) / len(accuracy_history)

        if avg_rolling_accuracy >= 0.85 and steps_since_graduation >= 300:
            steps_since_graduation = 0
            batch.graduate()

def infer(checkpoint_path, difficulty_str, turns = 1, verbose = False):
    # Setup device - CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Config
    config = HRMACTModelConfig(
        seq_len=81, vocab_size=10, high_level_cycles=2, low_level_cycles=2,
        transformers=HRMACTModelConfig.TransformerConfig(num_layers=4, hidden_size=256, num_heads=4, expansion=4),
        act=HRMACTModelConfig.ACTConfig(halt_max_steps=16, halt_exploration_probability=0.1)
    )

    # Load model
    model = HRMACTInner(config)
    
    # Load checkpoint with proper device handling
    try:
        # Try to load with the current device first
        model.load_state_dict(load_file(checkpoint_path, device=str(device)))
    except Exception as e:
        print(f"Error loading with {device}, trying CPU instead: {e}")
        # Fall back to CPU for loading, then move to target device
        model.load_state_dict(load_file(checkpoint_path, device="cpu"))
    
    # Move model to device after loading
    model = model.to(device, dtype=config.dtype)
    model.eval()
    print("Loaded model from checkpoint!")

    # Generate Sudoku
    difficulty_map = {
        "very-easy": Difficulty.VERY_EASY, "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD, "extreme": Difficulty.EXTREME
    }
    wins = 0
    for turn in range(turns):
        difficulty = difficulty_map[difficulty_str]
        puzzle, solution = generate_sudoku(difficulty)

        if verbose:
            print("Puzzle:\n", sudoku_board_string(puzzle))
            print("Solution:\n", sudoku_board_string(solution))

        puzzle_tensor = torch.tensor(puzzle.flatten(), dtype=torch.long, device=device).unsqueeze(0)

        # Initial hidden states
        low_h = model.initial_low_level.unsqueeze(0).expand(1, config.seq_len + 1, -1)
        high_h = model.initial_high_level.unsqueeze(0).expand(1, config.seq_len + 1, -1)
        hidden_states = (low_h, high_h)

        with torch.no_grad():
            for segment in range(1, config.act.halt_max_steps + 1):
                print(f"\n--- Segment {segment} ---")

                output = model(hidden_states, puzzle_tensor)
                hidden_states, output_logits, q_halt, q_continue = output

                predictions = output_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                # Display prediction
                accurate_squares, predicted_squares = 0, 0
                predicted_board_flat = []

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
                print(f"Predicted solution ({accurate_squares} / {predicted_squares}):")
                if verbose:
                    print(sudoku_board_string(predicted_board))

                q_h, q_c = torch.sigmoid(q_halt).item(), torch.sigmoid(q_continue).item()
                print(f"Q (halt - continue): {q_h:.4f} - {q_c:.4f}")

                if q_h > q_c:
                    print("Model halted.")
                    break
        win = np.allclose(predicted_board, solution)
        wins += 1 if win else 0
        if verbose:
            print(f"Game {turn}: {'Win' if win else 'Loss'}")
    print(f"Total wins: {wins}/{turns}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Reasoning Model in PyTorch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument("--use-sapient", action="store_true",
                              help="Use sapientinc/sudoku-extreme dataset instead of Ritvik19")

    # Infer command
    parser_infer = subparsers.add_parser("infer", help="Run inference with a trained model")
    parser_infer.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.safetensors)")
    parser_infer.add_argument("difficulty", type=str, choices=["very-easy", "easy", "medium", "hard", "extreme"],
                              help="Difficulty of the Sudoku puzzle to generate")
    parser_infer.add_argument("--turns", type=int, default=1, help="Number of turns (inference iterations) to run")
    parser_infer.add_argument("--verbose", action="store_true", help="Whether to print verbose output")

    args = parser.parse_args()

    if args.command == "train":
        train(use_sapient=getattr(args, 'use_sapient', False))
    elif args.command == "infer":
        infer(args.checkpoint, args.difficulty, args.turns, args.verbose)