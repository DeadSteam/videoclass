"""
Data models for video classification
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ClassificationResult:
    """Single classification result"""
    video_id: str
    true_label: str
    predicted_label: str
    confidence: float
    top_k_predictions: List[Tuple[str, float]]


@dataclass 
class ExperimentMetrics:
    """Metrics for an experiment"""
    accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_names: List[str]
    total_samples: int
    model_name: str

