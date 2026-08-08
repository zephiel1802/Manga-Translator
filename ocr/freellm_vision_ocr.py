"""
FreeLLM Vision OCR module.
Uses FreeLLM API (OpenAI-compatible) with vision capabilities to read text from images.
Best accuracy for vertical CJK text since LLMs understand reading direction.
"""
import base64
import os
import io
import time
import concurrent.futures
from PIL import Image
import numpy as np
from openai import OpenAI


class FreeLLMVisionOCR:
    """
    OCR engine using FreeLLM Vision API.
    
    Sends bubble crop images to FreeLLM with a prompt to read text.
    Much more accurate than Chrome Lens for vertical CJK text because
    the LLM understands reading direction and context.
    """
    
    # Language-specific prompts for optimal OCR
    LANG_PROMPTS = {
        "zh": "Read ALL Chinese text in this image. For vertical columns, read right-to-left, top-to-bottom. Return ONLY the raw text, nothing else. No explanations.",
        "ja": "Read ALL Japanese text in this image. For vertical columns, read right-to-left, top-to-bottom. Return ONLY the raw text, nothing else. No explanations.",
        "ko": "Read ALL Korean text in this image. Return ONLY the raw text, nothing else. No explanations.",
        "en": "Read ALL English text in this image. Return ONLY the raw text, nothing else. No explanations.",
    }
    
    DEFAULT_PROMPT = "Read ALL text in this image. Return ONLY the raw text, nothing else. No explanations."
    
    def __init__(self, api_key: str = None, base_url: str = None, ocr_language: str = "zh"):
        """
        Initialize FreeLLM Vision OCR.
        
        Args:
            api_key: FreeLLM API key
            base_url: FreeLLM API base URL
            ocr_language: BCP 47 language code (default: "zh")
        """
        self.api_key = api_key or os.environ.get("FREELLM_API_KEY")
        self.base_url = base_url or os.environ.get("FREELLM_BASE_URL", "http://127.0.0.1:31415/v1")
        self.ocr_language = ocr_language
        
        if not self.api_key:
            raise ValueError("FreeLLM API key required for Vision OCR.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0
        )
        self.model = "auto"
        print(f"[FreeLLM Vision OCR] Initialized (lang={ocr_language})")
    
    def _image_to_base64(self, image) -> str:
        """Convert PIL Image or numpy array to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _get_prompt(self) -> str:
        """Get language-specific OCR prompt."""
        return self.LANG_PROMPTS.get(self.ocr_language, self.DEFAULT_PROMPT)
    
    def __call__(self, image) -> str:
        """
        OCR a single image. Compatible with MangaOCR interface.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            str: Extracted text
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        base64_img = self._image_to_base64(image)
        prompt = self._get_prompt()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            # Clean up common LLM artifacts
            text = text.replace("```", "").strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"[FreeLLM Vision OCR] Error: {e}")
            return ""
    
    def process_batch(self, images: list) -> list:
        """
        Process multiple images sequentially (to avoid rate limits).
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extracted texts in same order
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images):
            if i > 0 and i % 5 == 0:
                print(f"    Vision OCR progress: {i}/{total}")
            
            text = self(img)
            results.append(text)
            
            # Small delay between requests to avoid rate limiting
            if i < total - 1:
                time.sleep(0.3)
        
        return results
