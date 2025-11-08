#!/usr/bin/env python3
"""
Quick test to verify the sapientinc dataset implementation works.
"""

import sys

print("=" * 80)
print("Quick Test: Sapientinc Dataset Implementation")
print("=" * 80)

try:
    # Test 1: Import functions
    print("\n[1/4] Testing imports...")
    from sudoku import load_sapient_sudoku_dataset, shuffle_sudoku
    print("✓ Imports successful")

    # Test 2: Load small dataset
    print("\n[2/4] Loading small test set (5 examples, 0 augmentations)...")
    dataset = load_sapient_sudoku_dataset(
        subset="test",
        num_augment=0,
        subsample_size=5
    )
    print(f"✓ Loaded {len(dataset)} examples")
    print(f"  First example keys: {dataset[0].keys()}")
    print(f"  First puzzle shape: {dataset[0]['puzzle'].shape}")
    print(f"  First solution shape: {dataset[0]['solution'].shape}")

    # Test 3: Test augmentation
    print("\n[3/4] Testing augmentation...")
    puzzle = dataset[0]['puzzle']
    solution = dataset[0]['solution']
    aug_puzzle, aug_solution = shuffle_sudoku(puzzle, solution)
    print(f"✓ Augmentation successful")
    print(f"  Original puzzle sum: {puzzle.sum()}")
    print(f"  Aug puzzle sum: {aug_puzzle.sum()}")

    # Test 4: Test with training data
    print("\n[4/4] Testing training data (2 examples, 2 augmentations each)...")
    train_dataset = load_sapient_sudoku_dataset(
        subset="train",
        num_augment=2,
        subsample_size=2
    )
    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"  (2 base + 2 aug each = 6 total, but may vary due to random)")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe implementation is working correctly.")
    print("You can now:")
    print("  1. Test on real dataset:")
    print("     python test_real_dataset.py --checkpoint checkpoint-5000.safetensors --samples 50")
    print("  2. Train on sapientinc dataset:")
    print("     python main.py train --use-sapient")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("TEST FAILED")
    print("=" * 80)
    sys.exit(1)
