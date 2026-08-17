import os
import json
import base64
import time
import cv2
import numpy as np
from PIL import Image
import io
import concurrent.futures
import threading
import logging

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai.types import HttpOptions
except ImportError:
    pass

DEFAULT_MODEL = "gemini-2.5-flash-image"
ANALYSIS_MODEL = "gemini-2.5-flash"
DEFAULT_CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_vision_credentials.json")
MAX_RETRIES = 5
RETRY_DELAY_BASE = 10
MAX_WORKERS = 3

LANG_NAMES = {
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-TW": "Traditional Chinese",
    "ko": "Korean",
    "en": "English",
    "vi": "Vietnamese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese"
}

VIETNAMESE_PROMPT_RULES = """
- Fix any OCR errors from the source text.
- Use natural spoken dialogue style for manga.
- Use proper Vietnamese pronouns based on context (tao/mày for enemies/close friends, tôi/anh/cô/chị/em, vv.).
- Keep character names in their original form.
- Translate exclamations appropriately (e.g., くそ -> Chết tiệt, やばい -> Toang rồi, vv.).
- Translate Sound Effects (SFX) naturally into Vietnamese or provide suitable equivalents.
"""

def cv2_to_pil(image_cv: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL Image."""
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def cv2_to_base64(image_cv: np.ndarray) -> str:
    """Convert OpenCV BGR image to base64 JPEG string."""
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_cv2(b64_str: str) -> np.ndarray:
    """Convert base64 image string to OpenCV BGR numpy array."""
    image_data = base64.b64decode(b64_str)
    pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_np = np.array(pil_image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

class BatchWorkerPool:
    """Thread pool with semaphore-based concurrency control and exponential backoff retry."""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.semaphore = threading.Semaphore(max_workers)
        
    def _execute_with_retry(self, func, *args, **kwargs):
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                    if retries == MAX_RETRIES:
                        raise e
                    delay = RETRY_DELAY_BASE * (2 ** retries)
                    time.sleep(delay)
                    retries += 1
                else:
                    raise e
                    
    def map_tasks(self, func, tasks_args, progress_callback=None):
        results = [None] * len(tasks_args)
        
        def worker_wrapper(index, args):
            with self.semaphore:
                res = self._execute_with_retry(func, *args)
                results[index] = res
                if progress_callback:
                    progress_callback(index + 1, len(tasks_args), f"Completed task {index + 1}/{len(tasks_args)}")
                return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(worker_wrapper, i, arg) for i, arg in enumerate(tasks_args)]
            concurrent.futures.wait(futures)
            
        return results

class GeminiMangaPipeline:
    """CORE module for manga translation app using Gemini API."""
    
    def __init__(self, api_key=None, project_id=None, location="us-central1", strategy="full", model=DEFAULT_MODEL, temperature=0.1, max_workers=MAX_WORKERS):
        """
        Initialize Gemini Manga Pipeline.
        
        Auth priority:
          1. api_key parameter (Google AI Studio)
          2. GEMINI_API_KEY environment variable
          3. Service Account JSON (google_vision_credentials.json → Vertex AI)
        
        Args:
            api_key: Gemini API key. If None, tries env var, then service account.
            project_id: GCP project ID (auto-detected from credentials JSON).
            location: GCP region for Vertex AI.
            strategy: "full" (Gemini does everything) or "hybrid" (Gemini analyzes, programmatic render).
            model: Gemini model ID for image generation.
            temperature: Low (0.0-0.2) for consistent output.
            max_workers: Max concurrent API calls for batch processing.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.project_id = project_id
        self.location = location
        self.strategy = strategy
        self.model = model
        self.temperature = temperature
        self.worker_pool = BatchWorkerPool(max_workers=max_workers)
        
        # Auto-detect project_id from credentials JSON
        if not self.project_id and os.path.exists(DEFAULT_CREDENTIALS_PATH):
            try:
                with open(DEFAULT_CREDENTIALS_PATH, 'r') as f:
                    creds = json.load(f)
                    self.project_id = creds.get('project_id')
            except Exception:
                pass
                
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            print("✓ Gemini Pipeline: Authenticated with Gemini API key")
        else:
            if not self.project_id:
                raise ValueError(
                    "Gemini Pipeline: No credentials found.\n"
                    "  Option 1: Enter Gemini API key in web form\n"
                    "  Option 2: Place google_vision_credentials.json in project root"
                )
            # Vertex AI - use v1beta1 for image generation/editing support
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                http_options=HttpOptions(api_version="v1beta1")
            )
            print(f"✓ Gemini Pipeline: Authenticated with Vertex AI v1beta1 (project={self.project_id})")

    def _build_full_prompt(self, source_lang: str, target_lang: str, context: str = "") -> str:
        """Build the translation prompt for Strategy A (Gemini Banana image editing)."""
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        
        # Short, direct prompt - image models work better with concise instructions
        prompt = (
            f"This is a manga/comic page in {source_name}. "
            f"Translate ALL text to {target_name}. "
            f"For each text area: erase the original {source_name} text completely, "
            f"reconstruct the background behind it, "
            f"then typeset the {target_name} translation in the same position. "
            f"Keep all artwork, characters, panels, and effects unchanged. "
            f"Return the fully translated page as an image."
        )
        
        # Add language-specific hints (keep short)
        if target_lang == "vi":
            prompt += (
                " Vietnamese rules: use natural spoken dialogue, "
                "proper pronouns (tao/mày, tôi/anh/chị/em), "
                "keep character names unchanged, "
                "translate SFX naturally."
            )
        
        if context:
            prompt += f" Story context: {context}"
        
        return prompt

    def translate_page_full(self, image_cv: np.ndarray, source_lang: str, target_lang: str, context: str = "") -> np.ndarray:
        """
        Strategy A: Send image + detailed prompt to Gemini Banana → get translated image.
        
        Args:
            image_cv: Input manga page as OpenCV BGR numpy array
            source_lang: Source language code (ja, zh, ko, en)
            target_lang: Target language code (vi, en, etc.)
            context: Optional story context from previous pages
            
        Returns:
            Translated image as OpenCV BGR numpy array (same dimensions as input)
        """
        prompt = self._build_full_prompt(source_lang, target_lang, context)
        
        # Convert OpenCV image to PIL Image (the format google-genai SDK accepts)
        pil_image = cv2_to_pil(image_cv)
        original_size = (image_cv.shape[1], image_cv.shape[0])  # (width, height)
        
        print(f"[Gemini Pipeline] Image size: {original_size[0]}x{original_size[1]}")
        print(f"[Gemini Pipeline] Model: {self.model}")
        print(f"[Gemini Pipeline] Has interactions API: {hasattr(self.client, 'interactions')}")
        
        # 1. Try NEW Interactions API (google-genai >= 1.x)
        if hasattr(self.client, 'interactions'):
            try:
                print("[Gemini Pipeline] Trying Interactions API...")
                b64_image = cv2_to_base64(image_cv)
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=[
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": b64_image, "mime_type": "image/jpeg"}
                    ]
                )
                if hasattr(interaction, 'output_image') and interaction.output_image and interaction.output_image.data:
                    result = base64_to_cv2(interaction.output_image.data)
                    if result.shape[:2] != image_cv.shape[:2]:
                        result = cv2.resize(result, original_size, interpolation=cv2.INTER_LANCZOS4)
                    print(f"[Gemini Pipeline] ✅ Interactions API success! Output: {result.shape[1]}x{result.shape[0]}")
                    return result
                else:
                    print("[Gemini Pipeline] Interactions API returned no output_image")
            except AttributeError:
                print("[Gemini Pipeline] Interactions API not available in this SDK version")
            except Exception as e:
                print(f"[Gemini Pipeline] Interactions API error: {e}")
                
        # 2. generate_content API with response_modalities
        try:
            from google.genai import types
            
            print(f"[Gemini Pipeline] Trying generate_content API...")
            print(f"[Gemini Pipeline] Prompt ({len(prompt)} chars): {prompt[:150]}...")
            
            # Convert image to bytes for explicit Part construction
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="JPEG", quality=95)
            img_bytes = img_buffer.getvalue()
            print(f"[Gemini Pipeline] Image bytes: {len(img_bytes)}")
            
            # Build contents with explicit types (not relying on SDK auto-conversion)
            contents = types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=img_bytes
                        )
                    ),
                    types.Part(text=prompt),
                ]
            )
            
            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=self.temperature,
            )
            print("[Gemini Pipeline] Using explicit types.Content + types.Part + types.Blob")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            # Debug: inspect response structure
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    print(f"[Gemini Pipeline] Response has {len(candidate.content.parts)} parts")
                    for i, part in enumerate(candidate.content.parts):
                        has_text = hasattr(part, 'text') and part.text
                        has_inline = hasattr(part, 'inline_data') and part.inline_data
                        print(f"[Gemini Pipeline]   Part {i}: text={bool(has_text)}, inline_data={bool(has_inline)}")
                        
                        if has_inline:
                            part_data = part.inline_data.data
                            mime = getattr(part.inline_data, 'mime_type', 'unknown')
                            print(f"[Gemini Pipeline]   Part {i}: mime={mime}, data_type={type(part_data).__name__}, size={len(part_data) if part_data else 0}")
                            
                            if isinstance(part_data, bytes):
                                result_pil = Image.open(io.BytesIO(part_data)).convert('RGB')
                                result = pil_to_cv2(result_pil)
                            elif isinstance(part_data, str):
                                result = base64_to_cv2(part_data)
                            else:
                                print(f"[Gemini Pipeline]   ⚠️ Unknown data type: {type(part_data)}")
                                continue
                            
                            if result.shape[:2] != image_cv.shape[:2]:
                                result = cv2.resize(result, original_size, interpolation=cv2.INTER_LANCZOS4)
                            print(f"[Gemini Pipeline] ✅ generate_content success! Output: {result.shape[1]}x{result.shape[0]}")
                            return result
                        
                        if has_text:
                            text_preview = part.text[:200] if part.text else ""
                            print(f"[Gemini Pipeline]   Part {i} text: {text_preview}")
                else:
                    print(f"[Gemini Pipeline] ⚠️ Response candidate has no content/parts")
                    # Check for safety/finish_reason
                    if hasattr(candidate, 'finish_reason'):
                        print(f"[Gemini Pipeline]   finish_reason: {candidate.finish_reason}")
            else:
                print("[Gemini Pipeline] ⚠️ Response has no candidates")
                if hasattr(response, 'prompt_feedback'):
                    print(f"[Gemini Pipeline]   prompt_feedback: {response.prompt_feedback}")
            
            print("[Gemini Pipeline] ❌ No image found in response")
        except Exception as e:
            print(f"[Gemini Pipeline] ❌ generate_content error: {e}")
            import traceback
            traceback.print_exc()
                
        # Return original image if both methods fail
        print("[Gemini Pipeline] ⚠️ Returning original image (no translation applied)")
        return image_cv
        
    def analyze_page(self, image_cv: np.ndarray, source_lang: str, target_lang: str) -> list:
        """Strategy B (Hybrid): Use Gemini Vision to analyze ALL text on page and translate.
        
        Returns list of text blocks with coords, translations, and metadata.
        Works on Vertex AI v1 (text model, uses $300 GCP credit).
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        
        h, w = image_cv.shape[:2]
        
        prompt = f"""Analyze this manga/comic page ({w}x{h} pixels) thoroughly.

TASK: Find ALL {source_name} text and translate to {target_name}.

You MUST detect EVERY type of text on the page:
- Dialog in speech bubbles
- Narration boxes (rectangular panels with story text)  
- Vertical text (read top-to-bottom, common in CJK manga)
- Sound effects / SFX / onomatopoeia (stylized text integrated into artwork)
- Small text, footnotes, page numbers
- Text outside bubbles (floating text near characters)

IMPORTANT COORDINATE RULES:
- "coords" must be the EXACT pixel bounding box [x1, y1, x2, y2] of where the ORIGINAL text appears
- Coordinates are in pixels relative to {w}x{h} image
- Be PRECISE - the coords will be used to erase the original text and place the translation
- For text inside speech bubbles, the coords should cover the TEXT AREA inside the bubble (not the whole bubble border)

For EACH text block found, return a JSON object with these EXACT keys:
- "coords": [x1, y1, x2, y2] - precise bounding box in pixels
- "original_text": the original {source_name} text exactly as written
- "translated_text": natural {target_name} translation
- "text_type": one of "dialog", "narration", "sfx", "vertical", "aside"
- "is_vertical": true if original text reads top-to-bottom, false if horizontal
- "text_color": "black" or "white" (based on background behind text)
- "font_weight": "normal", "bold", or "heavy" (for SFX use "heavy")
- "font_size_ratio": a float 0.0-1.0 indicating relative text size (1.0 = largest text on page, 0.3 = small text)

Return ONLY a JSON array. No markdown, no explanation."""

        # Vietnamese translation quality rules
        if target_lang == "vi":
            prompt += """

VIETNAMESE TRANSLATION RULES:
- Use natural spoken Vietnamese, NOT textbook style
- Dialog must sound like real conversation when read aloud
- Pronouns: pick appropriate level (tao/mày for rough, tôi/anh for polite, etc.)
- Keep character names in original form (don't translate names)
- SFX: translate to Vietnamese equivalents (ドーン→BÙM, ザー→RÀO RÀO)
- Historical terms: use Sino-Vietnamese where appropriate (皇帝→Hoàng đế, 大人→Đại nhân)
- Short impactful lines stay short"""

        print(f"[Gemini Hybrid] Analyzing page {w}x{h}, {source_name}→{target_name}")
        
        # Convert to PIL Image
        pil_image = cv2_to_pil(image_cv)
        
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=ANALYSIS_MODEL,
                contents=[pil_image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            
            if isinstance(result, list):
                # Validate and clamp coordinates
                validated = []
                for block in result:
                    coords = block.get('coords', [])
                    if len(coords) == 4:
                        x1, y1, x2, y2 = [int(c) for c in coords]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            block['coords'] = [x1, y1, x2, y2]
                            validated.append(block)
                
                print(f"[Gemini Hybrid] Found {len(validated)} text blocks:")
                for i, b in enumerate(validated):
                    tt = b.get('text_type', '?')
                    orig = b.get('original_text', '')[:25]
                    trans = b.get('translated_text', '')[:25]
                    vert = '↕' if b.get('is_vertical') else '↔'
                    print(f"  [{i+1}] {tt} {vert}: {orig} → {trans}")
                
                return validated
            else:
                print(f"[Gemini Hybrid] Unexpected result type: {type(result)}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"[Gemini Hybrid] JSON parse error: {e}")
            text = response.text if response else ""
            print(f"[Gemini Hybrid] Raw response: {text[:500]}")
            # Try to extract JSON array from text
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            return []
        except Exception as e:
            print(f"[Gemini Hybrid] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def process_batch(self, images_data: list, source_lang: str, target_lang: str, progress_callback=None) -> list:
        """Process a batch of images concurrently."""
        if self.strategy == "full":
            func = self.translate_page_full
        else:
            func = self.analyze_page
            
        # Unpack arguments for the worker pool
        tasks_args = [(img, source_lang, target_lang) for img in images_data]
        return self.worker_pool.map_tasks(func, tasks_args, progress_callback)
