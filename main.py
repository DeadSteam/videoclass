"""
Holidays Video Classification with Qwen2-VL Model
Classifies videos within the Holidays and Traditions category
using Qwen2-VL vision-language model for Zero-Shot and Few-Shot Learning

Main entry point for the classification system.
"""

import argparse
import torch

from experiment import run_experiment, save_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Holidays Classification with Qwen2-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       choices=["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"],
                       help="Qwen2-VL model variant")
    parser.add_argument("--n_samples", type=int, default=80, help="Number of test samples")
    parser.add_argument("--categories", type=int, default=8, help="Number of categories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  HOLIDAYS VIDEO CLASSIFICATION WITH QWEN2-VL")
    print("  Dataset: HowTo100M | Category: Holidays and Traditions")
    print(f"  Model: {args.model}")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected! Using CPU (will be slow)")
    
    # Run experiment
    metrics, results = run_experiment(
        model_name=args.model,
        n_samples=args.n_samples,
        top_categories=args.categories,
        random_state=args.seed
    )
    
    # Save results
    save_results(metrics, results)
    
    print("\n[OK] Experiment completed!")


if __name__ == "__main__":
    main()
