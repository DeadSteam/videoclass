"""
Dataset loader for Holidays and Traditions category from HowTo100M
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image


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
        """Get subcategories with at least 50 videos"""
        counts = self.holidays_df['category_2'].value_counts()
        return [cat for cat in counts.index if counts[cat] >= 50]
    
    def sample_for_classification(
        self, subcategories: Optional[List[str]] = None,
        n_samples: int = 100, n_shot: int = 5, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample support and query sets for few-shot classification
        
        Args:
            subcategories: List of subcategories to use (None = auto-select top 8)
            n_samples: Number of query samples per category
            n_shot: Number of support samples per category
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (support_samples_df, query_samples_df)
        """
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
        """
        Download and cache YouTube video thumbnail
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            PIL Image or None if download failed
        """
        hash_prefix = hashlib.md5(video_id.encode()).hexdigest()[:2]
        cache_path = self.thumbnail_cache_dir / hash_prefix / f"{video_id}.jpg"
        cache_path.parent.mkdir(exist_ok=True)
        
        # Check cache first
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except:
                pass
        
        # Try different YouTube thumbnail URLs
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

