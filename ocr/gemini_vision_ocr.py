"""
Gemini Vision OCR module.
Uses Google Gemini API to read text from images.
Extremely accurate for CJK text and can even read highly stylized sound effects.
"""
import io
import os
import json
from PIL import Image
import numpy as np
from google import genai
from google.genai.types import HttpOptions

DEFAULT_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "google_vision_credentials.json"
)

class GeminiVisionOCR:
    """
    OCR engine using Google Gemini Vision capabilities.
    """
    
    LANG_PROMPTS = {
        "zh": "Read ALL Chinese text in this image. For vertical columns, read right-to-left, top-to-bottom. Return ONLY the raw text, nothing else. No explanations.",
        "ja": "Read ALL Japanese text in this image. For vertical columns, read right-to-left, top-to-bottom. Return ONLY the raw text, nothing else. No explanations.",
        "ko": "Read ALL Korean text in this image. Return ONLY the raw text, nothing else. No explanations.",
        "en": "Read ALL English text in this image. Return ONLY the raw text, nothing else. No explanations.",
    }
    
    DEFAULT_PROMPT = "Read ALL text in this image. Return ONLY the raw text, nothing else. No explanations."
    
    def __init__(self, api_key: str = None, ocr_language: str = "zh"):
        """
        Initialize Gemini Vision OCR.
        
        Args:
            api_key: Gemini API key
            ocr_language: BCP 47 language code (default: "zh")
        """
        self.ocr_language = ocr_language
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = "gemini-flash-latest"
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            print("✓ Gemini Vision OCR: Authenticated with API key")
        else:
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path and os.path.exists(DEFAULT_CREDENTIALS_PATH):
                credentials_path = DEFAULT_CREDENTIALS_PATH
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            if not credentials_path or not os.path.exists(credentials_path):
                raise ValueError(
                    "Gemini Vision auth required. Provide API key or Service Account JSON."
                )
            
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
            project_id = creds_data.get("project_id")
            if not project_id:
                raise ValueError(f"No project_id found in {credentials_path}")
            
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location="us-central1",
                http_options=HttpOptions(api_version="v1")
            )
            print(f"✓ Gemini Vision OCR: Authenticated with Service Account (project={project_id})")
        
        print(f"[Gemini Vision OCR] Initialized (lang={ocr_language})")

    def _get_prompt(self) -> str:
        """Get language-specific OCR prompt."""
        return self.LANG_PROMPTS.get(self.ocr_language, self.DEFAULT_PROMPT)

    def __call__(self, image) -> str:
        """
        OCR a single image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        prompt = self._get_prompt()
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image]
            )
            text = response.text.strip()
            # Clean up potential markdown formatting
            text = text.replace("```", "").strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            print(f"[Gemini Vision OCR] Error: {e}")
            return ""

    def process_batch(self, images: list) -> list:
        """
        Process multiple images sequentially.
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images):
            if i > 0 and i % 5 == 0:
                print(f"    Gemini OCR progress: {i}/{total}")
            
            text = self(img)
            results.append(text)
            
        return results

    def detect_and_recognize_blocks(self, image, target_lang="vi") -> list:
        """
        End-to-End detection and translation using Gemini.
        Returns a list of dicts: {"coords": (x1, y1, x2, y2), "text": "...", "translated_text": "..."}
        """
        import re
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        w, h = image.size
            
        prompt = f"""You are an expert manga/comic translator.
Analyze this entire image and find all text blocks (both inside speech bubbles and outside stylized text).
Translate the text to {target_lang}.

Return a JSON array. Each object in the array must have:
- "original_text": The raw text you extracted.
- "translated_text": The translated text. Auto-correct any OCR errors before translating.
- "box_2d": The bounding box of the text block as an array [ymin, xmin, ymax, xmax] with values scaled from 0 to 1000.

Return ONLY the JSON array, nothing else. Example format:
[
  {{"original_text": "Hello", "translated_text": "Xin chào", "box_2d": [100, 200, 150, 300]}}
]"""

        try:
            # We use gemini-pro-latest for better spatial understanding, though flash might work
            response = self.client.models.generate_content(
                model="gemini-pro-latest",
                contents=[prompt, image]
            )
            text = response.text.strip()
            
            # Extract JSON block if it's wrapped in markdown
            json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = text
                
            blocks_data = json.loads(json_str)
            
            blocks = []
            for item in blocks_data:
                ymin, xmin, ymax, xmax = item["box_2d"]
                
                # Convert 0-1000 scale to absolute pixels
                x1 = int((xmin / 1000.0) * w)
                y1 = int((ymin / 1000.0) * h)
                x2 = int((xmax / 1000.0) * w)
                y2 = int((ymax / 1000.0) * h)
                
                blocks.append({
                    "coords": (x1, y1, x2, y2),
                    "text": item.get("original_text", ""),
                    "translated_text": item.get("translated_text", "")
                })
                
            return blocks
            
        except Exception as e:
            print(f"[Gemini Vision OCR] Full-page E2E Error: {e}")
            return []
