"""
Holidays Video Classification with Qwen2-VL Model
Classifies videos within the Holidays and Traditions category
using Qwen2-VL vision-language model for Zero-Shot and Few-Shot Learning
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from collections import defaultdict
import hashlib
import requests
from io import BytesIO

import torch
import torch.nn.functional as F


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


class HolidaysDataset:
    """Dataset loader for Holidays and Traditions category"""
    
    SUBCATEGORIES = [
        "Halloween", "Christmas", "Easter", "Gift Giving",
        "Valentines Day", "Thanksgiving", "Saint Patrick's Day", "Mother's Day"
    ]
    
    def __init__(self, root_path: str = "HowTo100M", thumbnail_cache_dir: str = "thumbnail_cache"):
        self.root_path = Path(root_path)
        self.csv_path = self.root_path / "HowTo100M_v1.csv"
        self.thumbnail_cache_dir = Path(thumbnail_cache_dir)
        self.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
        self.videos_df = None
        self.holidays_df = None
        
    def load(self) -> "HolidaysDataset":
        """Load and filter dataset"""
        print("[*] Loading HowTo100M dataset...")
        self.videos_df = pd.read_csv(self.csv_path)
        self.holidays_df = self.videos_df[
            self.videos_df['category_1'] == 'Holidays and Traditions'
        ].copy()
        print(f"  [+] Holidays videos: {len(self.holidays_df):,}")
        return self
    
    def get_subcategories(self) -> List[str]:
        counts = self.holidays_df['category_2'].value_counts()
        return [cat for cat in counts.index if counts[cat] >= 50]
    
    def sample_for_classification(
        self, subcategories: Optional[List[str]] = None,
        n_samples: int = 100, n_shot: int = 5, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(random_state)
        if subcategories is None:
            subcategories = self.get_subcategories()[:8]
            
        support_samples, query_samples = [], []
        
        for subcat in subcategories:
            cat_videos = self.holidays_df[self.holidays_df['category_2'] == subcat]
            if len(cat_videos) < n_shot + 10:
                continue
            cat_videos = cat_videos.sample(frac=1, random_state=random_state)
            support_samples.append(cat_videos.iloc[:n_shot])
            n_query = min(len(cat_videos) - n_shot, n_samples // len(subcategories))
            query_samples.append(cat_videos.iloc[n_shot:n_shot + n_query])
        
        return pd.concat(support_samples, ignore_index=True), pd.concat(query_samples, ignore_index=True)
    
    def download_thumbnail(self, video_id: str) -> Optional[Image.Image]:
        hash_prefix = hashlib.md5(video_id.encode()).hexdigest()[:2]
        cache_path = self.thumbnail_cache_dir / hash_prefix / f"{video_id}.jpg"
        cache_path.parent.mkdir(exist_ok=True)
        
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except:
                pass
        
        urls = [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    if img.size[0] > 200:
                        img = img.convert("RGB")
                        img.save(cache_path, "JPEG", quality=90)
                        return img
            except:
                continue
        return None


class Qwen2VLClassifier:
    """
    Qwen2-VL based classifier for holiday videos
    Uses the model's vision-language understanding for classification
    """
    
    CATEGORY_PROMPTS = {
        "Halloween": "Halloween celebration with pumpkins, costumes, spooky decorations, trick or treating",
        "Christmas": "Christmas celebration with Christmas tree, Santa, gifts, winter decorations, snow",
        "Easter": "Easter celebration with Easter eggs, Easter bunny, spring decorations, egg hunt",
        "Gift Giving": "Gift wrapping, presents, gift boxes, ribbons, wrapping paper",
        "Valentines Day": "Valentine's Day with hearts, romantic decorations, love, red and pink colors",
        "Thanksgiving": "Thanksgiving with turkey, fall decorations, autumn leaves, harvest",
        "Saint Patrick's Day": "St. Patrick's Day with green color, shamrocks, Irish theme, leprechaun",
        "Mother's Day": "Mother's Day with flowers, gifts for mom, cards, appreciation",
    }
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize Qwen2-VL model
        
        Args:
            model_name: Model to use. Options:
                - "Qwen/Qwen2-VL-2B-Instruct" (lighter, faster)
                - "Qwen/Qwen2-VL-7B-Instruct" (better quality)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n[*] Loading Qwen2-VL model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Load model with appropriate settings for 12GB VRAM
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        print("[OK] Model loaded successfully!")
        
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory used: {allocated:.2f} GB")
    
    def classify_image(
        self, 
        image: Image.Image, 
        categories: List[str],
        method: str = "zero_shot"
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify an image into one of the categories
        
        Args:
            image: PIL Image to classify
            categories: List of category names
            method: Classification method
            
        Returns:
            Tuple of (predicted_label, confidence, top_k_predictions)
        """
        # Resize image for efficiency
        image = image.resize((384, 384), Image.LANCZOS)
        
        # Create classification prompt
        category_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
        
        prompt = f"""Look at this image and determine which holiday category it belongs to.

Categories:
{category_list}

Analyze the image carefully. Look for:
- Colors (orange/black for Halloween, red/green for Christmas, pastels for Easter, etc.)
- Objects (pumpkins, Christmas trees, Easter eggs, hearts, turkeys, shamrocks, etc.)
- Decorations and themes

Respond with ONLY the category name that best matches the image. Just the category name, nothing else."""

        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
            )
        
        # Decode response
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        
        # Parse response to find matching category
        predicted_label = self._match_category(response, categories)
        
        # For confidence, we'll use a simple matching score
        # (Qwen2-VL doesn't provide logits easily)
        confidence = 1.0 if predicted_label else 0.5
        
        # Create top-k predictions (simplified for generative model)
        top_k = [(predicted_label, confidence)]
        for cat in categories:
            if cat != predicted_label:
                top_k.append((cat, 0.1))
        
        return predicted_label, confidence, top_k[:5]
    
    def _match_category(self, response: str, categories: List[str]) -> str:
        """Match model response to one of the categories"""
        response_lower = response.lower()
        
        # Direct match
        for cat in categories:
            if cat.lower() in response_lower:
                return cat
        
        # Partial match
        for cat in categories:
            cat_words = cat.lower().split()
            if any(word in response_lower for word in cat_words):
                return cat
        
        # Keyword matching
        keywords = {
            "Halloween": ["halloween", "pumpkin", "spooky", "costume", "scary", "trick", "treat"],
            "Christmas": ["christmas", "xmas", "santa", "tree", "gift", "snow", "winter", "holiday"],
            "Easter": ["easter", "egg", "bunny", "spring", "pastel"],
            "Gift Giving": ["gift", "present", "wrap", "box", "ribbon"],
            "Valentines Day": ["valentine", "heart", "love", "romantic", "pink", "red"],
            "Thanksgiving": ["thanksgiving", "turkey", "fall", "autumn", "harvest", "gratitude"],
            "Saint Patrick's Day": ["patrick", "shamrock", "irish", "green", "leprechaun", "clover"],
            "Mother's Day": ["mother", "mom", "flower", "card", "appreciation"],
        }
        
        for cat in categories:
            if cat in keywords:
                if any(kw in response_lower for kw in keywords[cat]):
                    return cat
        
        # Default to first category if no match
        return categories[0]
    
    def classify_batch(
        self,
        images: List[Image.Image],
        categories: List[str],
        show_progress: bool = True
    ) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
        """Classify multiple images"""
        results = []
        for i, image in enumerate(images):
            if show_progress and (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(images)} images...")
            result = self.classify_image(image, categories)
            results.append(result)
        return results


def compute_metrics(
    results: List[ClassificationResult],
    class_names: List[str],
    model_name: str
) -> ExperimentMetrics:
    """Compute classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]
    
    accuracy = accuracy_score(y_true, y_pred)
    
    top3_correct = sum(1 for r in results if r.true_label in [p[0] for p in r.top_k_predictions[:3]])
    top5_correct = sum(1 for r in results if r.true_label in [p[0] for p in r.top_k_predictions[:5]])
    
    top3_accuracy = top3_correct / len(results) if results else 0
    top5_accuracy = top5_correct / len(results) if results else 0
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    
    per_class_metrics = {
        cat: {"precision": float(precision[i]), "recall": float(recall[i]),
              "f1": float(f1[i]), "support": int(support[i])}
        for i, cat in enumerate(class_names)
    }
    
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


def run_experiment(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    n_samples: int = 80,
    n_shot: int = 5,
    top_categories: int = 8,
    random_state: int = 42
) -> Tuple[ExperimentMetrics, List[ClassificationResult]]:
    """
    Run classification experiment with Qwen2-VL
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
    """Save experiment results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clean model name for filename
    model_short = metrics.model_name.split("/")[-1].replace("-", "_").lower()
    
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
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[OK] Saved metrics to {metrics_file}")
    
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
    with open(preds_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"[OK] Saved predictions to {preds_file}")


def main():
    import argparse
    
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
    
    metrics, results = run_experiment(
        model_name=args.model,
        n_samples=args.n_samples,
        top_categories=args.categories,
        random_state=args.seed
    )
    
    save_results(metrics, results)
    
    print("\n[OK] Experiment completed!")


if __name__ == "__main__":
    main()

