"""
Qwen2-VL based classifier for holiday videos
"""

from typing import List, Tuple, Optional
from PIL import Image

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


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
        method: str = "zero_shot",
        support_examples: Optional[List[Tuple[Image.Image, str]]] = None
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify an image into one of the categories
        
        Args:
            image: PIL Image to classify
            categories: List of category names
            method: Classification method ("zero_shot" or "few_shot")
            support_examples: List of (image, label) tuples for few-shot learning
            
        Returns:
            Tuple of (predicted_label, confidence, top_k_predictions)
        """
        # Resize image for efficiency
        image = image.resize((384, 384), Image.LANCZOS)
        
        # Create classification prompt
        category_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
        
        # Prepare messages for Qwen2-VL
        messages = []
        
        # Few-Shot Learning: Add support examples first
        if method == "few_shot" and support_examples:
            examples_text = "Here are some examples:\n\n"
            support_content = []
            
            for i, (support_img, support_label) in enumerate(support_examples):
                support_img_resized = support_img.resize((384, 384), Image.LANCZOS)
                support_content.append({"type": "image", "image": support_img_resized})
                examples_text += f"Example {i+1}: This is a {support_label} image.\n"
            
            # Add examples message
            messages.append({
                "role": "user",
                "content": support_content + [{"type": "text", "text": examples_text}]
            })
            
            # Add assistant response acknowledging examples
            messages.append({
                "role": "assistant",
                "content": "I understand. I'll use these examples to classify the new image."
            })
        
        # Main classification prompt
        if method == "few_shot" and support_examples:
            prompt = f"""Now classify this new image using the examples above as reference.

Categories:
{category_list}

Based on the examples, determine which holiday category this image belongs to. Look for similar:
- Colors (orange/black for Halloween, red/green for Christmas, pastels for Easter, etc.)
- Objects (pumpkins, Christmas trees, Easter eggs, hearts, turkeys, shamrocks, etc.)
- Decorations and themes

Respond with ONLY the category name that best matches the image. Just the category name, nothing else."""
        else:
            # Zero-Shot prompt
            prompt = f"""Look at this image and determine which holiday category it belongs to.

Categories:
{category_list}

Analyze the image carefully. Look for:
- Colors (orange/black for Halloween, red/green for Christmas, pastels for Easter, etc.)
- Objects (pumpkins, Christmas trees, Easter eggs, hearts, turkeys, shamrocks, etc.)
- Decorations and themes

Respond with ONLY the category name that best matches the image. Just the category name, nothing else."""

        # Add classification message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        })
        
        # Process input
        # For few-shot, we need to handle multiple images
        if method == "few_shot" and support_examples:
            # Collect all images (support + query)
            all_images = [img.resize((384, 384), Image.LANCZOS) for img, _ in support_examples]
            all_images.append(image)
        else:
            all_images = [image]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=all_images,
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

