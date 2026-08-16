"""
Gemini Inpainter - Text removal from manga pages using Gemini Banana.

Uses Gemini's native image editing capabilities to remove text from manga panels
and reconstruct the background art. Falls back to LaMa neural inpainting if
Gemini fails or is unavailable.

Usage:
    inpainter = GeminiInpainter(api_key="...", project_id="...")
    cleaned = inpainter.inpaint(image_cv, text_regions=[(x1,y1,x2,y2), ...])
"""
import os
import io
import json
import time
import base64
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

from google import genai
from google.genai.types import HttpOptions

# Path to service account JSON
DEFAULT_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "google_vision_credentials.json"
)

# Retry constants
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5

# Default model for inpainting
DEFAULT_MODEL = "gemini-2.5-flash-image"


class GeminiInpainter:
    """
    Removes text from manga pages using Gemini Banana's conversational image editing.
    
    Gemini Banana can understand natural language instructions to remove text
    and reconstruct the background - no pixel mask required.
    
    Falls back to LaMa inpainting if Gemini fails.
    """
    
    def __init__(
        self,
        api_key: str = None,
        project_id: str = None,
        location: str = "us-central1",
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize Gemini Inpainter.
        
        Auth priority:
          1. api_key parameter
          2. GEMINI_API_KEY environment variable
          3. Service Account JSON (google_vision_credentials.json)
        
        Args:
            api_key: Gemini API key
            project_id: GCP project ID (for Vertex AI auth)
            location: GCP region
            model: Model ID for image generation
        """
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Initialize Gemini client with dual auth
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            print("✓ Gemini Inpainter: Authenticated with API key")
        else:
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path and os.path.exists(DEFAULT_CREDENTIALS_PATH):
                credentials_path = DEFAULT_CREDENTIALS_PATH
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            if not credentials_path or not os.path.exists(credentials_path):
                raise ValueError(
                    "Gemini Inpainter auth required. Provide API key or Service Account JSON."
                )
            
            # Read project_id from credentials if not provided
            if not project_id:
                with open(credentials_path, 'r') as f:
                    creds_data = json.load(f)
                project_id = creds_data.get("project_id")
            
            if not project_id:
                raise ValueError(f"No project_id found in {credentials_path}")
            
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                http_options=HttpOptions(api_version="v1")
            )
            print(f"✓ Gemini Inpainter: Authenticated with Service Account (project={project_id})")
        
        # Lazy-load LaMa as fallback
        self._lama = None
    
    def _get_lama_fallback(self):
        """Lazy-load LaMa inpainter as fallback."""
        if self._lama is None:
            try:
                from lama_inpainter import get_lama_inpainter, LAMA_AVAILABLE
                if LAMA_AVAILABLE:
                    self._lama = get_lama_inpainter()
                    print("✓ Gemini Inpainter: LaMa fallback loaded")
                else:
                    self._lama = False  # Explicitly mark as unavailable
            except ImportError:
                self._lama = False
        return self._lama if self._lama is not False else None
    
    def _cv2_to_base64(self, image_cv: np.ndarray) -> str:
        """Convert OpenCV BGR image to base64 JPEG string."""
        import cv2
        _, buffer = cv2.imencode('.jpg', image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _base64_to_cv2(self, b64_data: str, target_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Convert base64 image data to OpenCV BGR numpy array.
        Optionally resize to match target_shape (height, width).
        """
        import cv2
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Resize to match original dimensions if needed
        if target_shape and img_cv is not None:
            h, w = target_shape[:2]
            if img_cv.shape[0] != h or img_cv.shape[1] != w:
                img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return img_cv
    
    def inpaint(
        self,
        image_cv: np.ndarray,
        text_regions: List[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Remove all text from a manga page using Gemini Banana.
        
        Args:
            image_cv: Input image as OpenCV BGR numpy array
            text_regions: Optional list of (x1, y1, x2, y2) bounding boxes
                         of text regions. If None, Gemini auto-detects text.
        
        Returns:
            Cleaned image as OpenCV BGR numpy array (same dimensions as input)
        """
        # Build region description for prompt (optional, helps Gemini focus)
        region_hint = ""
        if text_regions:
            h, w = image_cv.shape[:2]
            descriptions = []
            for i, (x1, y1, x2, y2) in enumerate(text_regions):
                # Describe position relative to image
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                pos = ""
                if cy < 0.33:
                    pos += "top "
                elif cy > 0.66:
                    pos += "bottom "
                else:
                    pos += "middle "
                if cx < 0.33:
                    pos += "left"
                elif cx > 0.66:
                    pos += "right"
                else:
                    pos += "center"
                descriptions.append(f"Region {i+1}: {pos.strip()}")
            region_hint = f"\nText regions to clean: {'; '.join(descriptions)}"
        
        prompt = (
            "This is a manga/comic page. Remove ALL text from this image completely. "
            "This includes: text in speech bubbles, narration boxes, sound effects (SFX/onomatopoeia), "
            "and any other text overlays. "
            "Reconstruct the background art behind the removed text seamlessly. "
            "Keep ALL artwork, characters, panel borders, screentones, and visual effects intact. "
            "The speech bubbles should remain but be empty (clean white/colored interior). "
            "Do NOT add any new text. Return only the cleaned image."
            f"{region_hint}"
        )
        
        # Try Gemini Banana first
        for attempt in range(MAX_RETRIES):
            try:
                b64_image = self._cv2_to_base64(image_cv)
                
                # Try new Interactions API first, fall back to legacy
                result_image = self._call_gemini(b64_image, prompt, image_cv.shape)
                
                if result_image is not None:
                    print(f"✓ Gemini Inpainter: Successfully cleaned image")
                    return result_image
                    
            except Exception as e:
                error_str = str(e)
                print(f"Gemini Inpainter attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"⚠️ Rate limit! Waiting {wait_time}s...")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait_time)
                        continue
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE)
                else:
                    break
        
        # Fallback to LaMa
        print("⚠️ Gemini Inpainter failed, falling back to LaMa...")
        return self._fallback_lama(image_cv, text_regions)
    
    def _call_gemini(
        self,
        b64_image: str,
        prompt: str,
        target_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Call Gemini API for image editing.
        Tries Interactions API first, falls back to generate_content.
        """
        # Method 1: Try new Interactions API (google-genai >= 1.x)
        try:
            if hasattr(self.client, 'interactions'):
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=[
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": b64_image, "mime_type": "image/jpeg"}
                    ]
                )
                
                if interaction.output_image and interaction.output_image.data:
                    return self._base64_to_cv2(interaction.output_image.data, target_shape)
        except AttributeError:
            pass  # interactions API not available, try legacy
        except Exception as e:
            print(f"  Interactions API error: {e}, trying legacy API...")
        
        # Method 2: Legacy generate_content with response_modalities
        try:
            img_pil = Image.open(io.BytesIO(base64.b64decode(b64_image)))
        except Exception:
            import cv2
            img_bytes = base64.b64decode(b64_image)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, img_pil],
            config={
                "response_modalities": ["IMAGE"],
                "temperature": 0.1,
            }
        )
        
        # Extract image from response
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return self._base64_to_cv2(
                        base64.b64encode(part.inline_data.data).decode('utf-8'),
                        target_shape
                    )
        
        return None
    
    def _fallback_lama(
        self,
        image_cv: np.ndarray,
        text_regions: List[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Fallback to LaMa inpainting or simple blur when Gemini fails.
        
        Args:
            image_cv: Input BGR image
            text_regions: List of (x1, y1, x2, y2) text bounding boxes
        
        Returns:
            Cleaned BGR image
        """
        import cv2
        
        lama = self._get_lama_fallback()
        
        if lama and text_regions:
            # Create a mask from text regions
            h, w = image_cv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in text_regions:
                # Expand region slightly for better inpainting
                pad = 5
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                mask[y1:y2, x1:x2] = 255
            
            # Dilate mask for better coverage
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            try:
                return lama.inpaint(image_cv, mask)
            except Exception as e:
                print(f"LaMa inpainting failed: {e}")
        
        # Ultimate fallback: OpenCV Telea inpainting
        if text_regions:
            h, w = image_cv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in text_regions:
                mask[y1:y2, x1:x2] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            return cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)
        
        # No regions specified - return original
        return image_cv.copy()


# Singleton accessor
_gemini_inpainter = None

def get_gemini_inpainter(api_key: str = None, **kwargs) -> GeminiInpainter:
    """Get or create singleton GeminiInpainter instance."""
    global _gemini_inpainter
    if _gemini_inpainter is None:
        try:
            _gemini_inpainter = GeminiInpainter(api_key=api_key, **kwargs)
        except Exception as e:
            print(f"⚠️ Could not initialize GeminiInpainter: {e}")
            return None
    return _gemini_inpainter
