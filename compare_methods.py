"""
Compare Zero-Shot vs Few-Shot Learning methods
"""

import json
from pathlib import Path
from experiment import run_experiment, save_results
from models import ExperimentMetrics


def compare_methods(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    n_samples: int = 80,
    n_shot: int = 5,
    top_categories: int = 8,
    random_state: int = 42
):
    """
    Run both Zero-Shot and Few-Shot experiments and compare results
    """
    print("\n" + "="*70)
    print("  COMPARING ZERO-SHOT vs FEW-SHOT LEARNING")
    print("="*70)
    
    results_comparison = {}
    
    # Run Zero-Shot experiment
    print("\n" + "="*70)
    print("  EXPERIMENT 1: ZERO-SHOT LEARNING")
    print("="*70)
    metrics_zero, results_zero = run_experiment(
        model_name=model_name,
        n_samples=n_samples,
        n_shot=n_shot,
        top_categories=top_categories,
        random_state=random_state,
        method="zero_shot"
    )
    
    # Save Zero-Shot results
    save_results(metrics_zero, results_zero, output_dir="results")
    results_comparison["zero_shot"] = {
        "accuracy": metrics_zero.accuracy,
        "top3_accuracy": metrics_zero.top3_accuracy,
        "top5_accuracy": metrics_zero.top5_accuracy,
        "macro_f1": metrics_zero.macro_f1,
        "macro_precision": metrics_zero.macro_precision,
        "macro_recall": metrics_zero.macro_recall,
    }
    
    # Run Few-Shot experiment
    print("\n" + "="*70)
    print("  EXPERIMENT 2: FEW-SHOT LEARNING")
    print("="*70)
    metrics_few, results_few = run_experiment(
        model_name=model_name,
        n_samples=n_samples,
        n_shot=n_shot,
        top_categories=top_categories,
        random_state=random_state,
        method="few_shot"
    )
    
    # Save Few-Shot results
    save_results(metrics_few, results_few, output_dir="results")
    results_comparison["few_shot"] = {
        "accuracy": metrics_few.accuracy,
        "top3_accuracy": metrics_few.top3_accuracy,
        "top5_accuracy": metrics_few.top5_accuracy,
        "macro_f1": metrics_few.macro_f1,
        "macro_precision": metrics_few.macro_precision,
        "macro_recall": metrics_few.macro_recall,
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Zero-Shot':<15} {'Few-Shot':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ("Accuracy", "accuracy"),
        ("Top-3 Accuracy", "top3_accuracy"),
        ("Top-5 Accuracy", "top5_accuracy"),
        ("Macro F1", "macro_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        zero_val = results_comparison["zero_shot"][metric_key]
        few_val = results_comparison["few_shot"][metric_key]
        diff = few_val - zero_val
        diff_pct = (diff / zero_val * 100) if zero_val > 0 else 0
        
        print(f"{metric_name:<25} {zero_val:>6.2%}      {few_val:>6.2%}      {diff:>+6.2%} ({diff_pct:+.1f}%)")
    
    # Per-class comparison
    print("\n" + "="*70)
    print("  PER-CLASS F1 SCORE COMPARISON")
    print("="*70)
    print(f"\n{'Category':<25} {'Zero-Shot F1':<15} {'Few-Shot F1':<15} {'Difference':<15}")
    print("-" * 70)
    
    for cat in metrics_zero.class_names:
        zero_f1 = metrics_zero.per_class_metrics[cat]["f1"]
        few_f1 = metrics_few.per_class_metrics[cat]["f1"]
        diff = few_f1 - zero_f1
        
        print(f"{cat:<25} {zero_f1:>6.2%}      {few_f1:>6.2%}      {diff:>+6.2%}")
    
    # Save comparison
    comparison_file = Path("results") / "zero_vs_few_shot_comparison.json"
    comparison_file.parent.mkdir(exist_ok=True)
    
    comparison_data = {
        "model": model_name,
        "n_samples": n_samples,
        "n_shot": n_shot,
        "categories": top_categories,
        "random_state": random_state,
        "zero_shot": results_comparison["zero_shot"],
        "few_shot": results_comparison["few_shot"],
        "per_class_comparison": {
            cat: {
                "zero_shot_f1": metrics_zero.per_class_metrics[cat]["f1"],
                "few_shot_f1": metrics_few.per_class_metrics[cat]["f1"],
                "difference": metrics_few.per_class_metrics[cat]["f1"] - metrics_zero.per_class_metrics[cat]["f1"]
            }
            for cat in metrics_zero.class_names
        }
    }
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Comparison saved to {comparison_file}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    accuracy_improvement = results_comparison["few_shot"]["accuracy"] - results_comparison["zero_shot"]["accuracy"]
    f1_improvement = results_comparison["few_shot"]["macro_f1"] - results_comparison["zero_shot"]["macro_f1"]
    
    print(f"\nFew-Shot улучшил Accuracy на: {accuracy_improvement:+.2%}")
    print(f"Few-Shot улучшил Macro F1 на: {f1_improvement:+.2%}")
    
    if accuracy_improvement > 0:
        print("\n[+] Few-Shot Learning показал лучшие результаты!")
    elif accuracy_improvement < 0:
        print("\n[!] Few-Shot Learning показал худшие результаты (возможно, нужна настройка)")
    else:
        print("\n[=] Результаты одинаковые")
    
    return metrics_zero, metrics_few, results_zero, results_few


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Zero-Shot vs Few-Shot")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Qwen2-VL model variant")
    parser.add_argument("--n_samples", type=int, default=80, help="Number of test samples")
    parser.add_argument("--categories", type=int, default=8, help="Number of categories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    compare_methods(
        model_name=args.model,
        n_samples=args.n_samples,
        top_categories=args.categories,
        random_state=args.seed
    )

