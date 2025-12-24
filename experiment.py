"""
Experiment execution and result saving
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from PIL import Image

from models import ClassificationResult, ExperimentMetrics
from dataset import HolidaysDataset
from classifier import Qwen2VLClassifier
from metrics import compute_metrics


def run_experiment(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    n_samples: int = 80,
    n_shot: int = 5,
    top_categories: int = 8,
    random_state: int = 42
) -> Tuple[ExperimentMetrics, List[ClassificationResult]]:
    """
    Run classification experiment with Qwen2-VL
    
    Args:
        model_name: Qwen2-VL model name
        n_samples: Number of test samples per category
        n_shot: Number of support samples per category
        top_categories: Number of categories to use
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (metrics, results)
    """
    print("\n" + "="*70)
    print("  HOLIDAYS CLASSIFICATION WITH QWEN2-VL")
    print(f"  Model: {model_name}")
    print(f"  Samples: {n_samples}")
    print("="*70 + "\n")
    
    # Load dataset
    dataset = HolidaysDataset()
    dataset.load()
    
    subcategories = dataset.get_subcategories()[:top_categories]
    print(f"\n[*] Using {len(subcategories)} subcategories:")
    for cat in subcategories:
        print(f"    - {cat}")
    
    # Sample data
    support_df, query_df = dataset.sample_for_classification(
        subcategories=subcategories,
        n_samples=n_samples,
        n_shot=n_shot,
        random_state=random_state
    )
    
    print(f"\n[*] Support set: {len(support_df)} samples")
    print(f"[*] Query set: {len(query_df)} samples")
    
    # Load Qwen2-VL model
    classifier = Qwen2VLClassifier(model_name=model_name)
    
    # Run classification
    print(f"\n[*] Classifying {len(query_df)} query samples...")
    results = []
    
    for idx, row in query_df.iterrows():
        video_id = row['video_id']
        true_label = row['category_2']
        
        # Download thumbnail
        thumbnail = dataset.download_thumbnail(video_id)
        
        if thumbnail is None:
            thumbnail = Image.new('RGB', (384, 384), color='gray')
        
        # Classify
        pred_label, conf, top_k = classifier.classify_image(thumbnail, subcategories)
        
        results.append(ClassificationResult(
            video_id=video_id,
            true_label=true_label,
            predicted_label=pred_label,
            confidence=conf,
            top_k_predictions=top_k
        ))
        
        if (len(results)) % 10 == 0:
            print(f"   Processed {len(results)}/{len(query_df)} samples...")
    
    # Compute metrics
    metrics = compute_metrics(results, subcategories, model_name)
    
    # Print results
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"\n  Model: {model_name}")
    print(f"\n  Accuracy:       {metrics.accuracy:.2%}")
    print(f"  Top-3 Accuracy: {metrics.top3_accuracy:.2%}")
    print(f"  Top-5 Accuracy: {metrics.top5_accuracy:.2%}")
    print(f"  Macro F1:       {metrics.macro_f1:.2%}")
    
    print("\n  Per-class F1 scores:")
    for cat, m in sorted(metrics.per_class_metrics.items(), key=lambda x: -x[1]['f1']):
        print(f"    {cat:25s}: {m['f1']:.2%} (n={m['support']})")
    
    return metrics, results


def save_results(
    metrics: ExperimentMetrics,
    results: List[ClassificationResult],
    output_dir: str = "results"
):
    """
    Save experiment results to JSON files
    
    Args:
        metrics: Experiment metrics
        results: List of classification results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clean model name for filename
    model_short = metrics.model_name.split("/")[-1].replace("-", "_").lower()
    
    # Save metrics
    metrics_dict = {
        "model": metrics.model_name,
        "accuracy": metrics.accuracy,
        "top3_accuracy": metrics.top3_accuracy,
        "top5_accuracy": metrics.top5_accuracy,
        "macro_f1": metrics.macro_f1,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "per_class": metrics.per_class_metrics,
        "confusion_matrix": metrics.confusion_matrix,
        "class_names": metrics.class_names,
        "total_samples": metrics.total_samples,
        "timestamp": datetime.now().isoformat()
    }
    
    metrics_file = output_path / f"holidays_qwen_{model_short}_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved metrics to {metrics_file}")
    
    # Save predictions
    predictions = [
        {
            "video_id": r.video_id,
            "true_label": r.true_label,
            "predicted_label": r.predicted_label,
            "confidence": r.confidence
        }
        for r in results
    ]
    
    preds_file = output_path / f"holidays_qwen_{model_short}_predictions.json"
    with open(preds_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved predictions to {preds_file}")

