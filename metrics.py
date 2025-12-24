"""
Metrics computation for classification experiments
"""

import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models import ClassificationResult, ExperimentMetrics


def compute_metrics(
    results: List[ClassificationResult],
    class_names: List[str],
    model_name: str
) -> ExperimentMetrics:
    """
    Compute classification metrics from results
    
    Args:
        results: List of classification results
        class_names: List of class names
        model_name: Name of the model used
        
    Returns:
        ExperimentMetrics object with all computed metrics
    """
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]
    
    # Top-1 accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Top-3 and Top-5 accuracy
    top3_correct = sum(1 for r in results if r.true_label in [p[0] for p in r.top_k_predictions[:3]])
    top5_correct = sum(1 for r in results if r.true_label in [p[0] for p in r.top_k_predictions[:5]])
    
    top3_accuracy = top3_correct / len(results) if results else 0
    top5_accuracy = top5_correct / len(results) if results else 0
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    
    per_class_metrics = {
        cat: {
            "precision": float(precision[i]), 
            "recall": float(recall[i]),
            "f1": float(f1[i]), 
            "support": int(support[i])
        }
        for i, cat in enumerate(class_names)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    return ExperimentMetrics(
        accuracy=accuracy,
        top3_accuracy=top3_accuracy,
        top5_accuracy=top5_accuracy,
        macro_f1=float(np.mean(f1)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        total_samples=len(results),
        model_name=model_name
    )

