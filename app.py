from flask import Flask, render_template, request, redirect, send_file, jsonify, url_for as flask_url_for
import builtins
import datetime

# Override print to include timestamps
original_print = builtins.print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    original_print(f"[{timestamp}]", *args, **kwargs)
builtins.print = timestamped_print

from flask_socketio import SocketIO, emit
import io
import zipfile
import json
import warnings
import os
import sys
import uuid
import time as time_module

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detect_bubbles import detect_bubbles
from process_bubble import process_bubble, process_bubble_auto, is_dark_bubble, get_bubble_background_color, get_dominant_color, process_bubble_preserve_gradient
from translator.translator import MangaTranslator

# PanelCleanerZ integration for text detection + cleaning
try:
    from pcleaner_bridge import get_pcleaner_bridge
    _pcleaner = get_pcleaner_bridge()
    PCLEANER_AVAILABLE = True
    print("PanelCleanerZ bridge loaded (Comic Text Detector + LaMa inpainting)")
except Exception as e:
    PCLEANER_AVAILABLE = False
    print(f"PanelCleanerZ not available, using fallback: {e}")

# Real-ESRGAN upscaler
try:
    from upscaler import get_manga_upscaler, MangaUpscaler
    UPSCALER_AVAILABLE = MangaUpscaler.is_available()
    if UPSCALER_AVAILABLE:
        print("Real-ESRGAN upscaler available")
    else:
        print("Real-ESRGAN not installed (pip install realesrgan) - upscale disabled")
except ImportError:
    UPSCALER_AVAILABLE = False
    print("Upscaler module not available")
    
try:
    from lama_inpainter import get_lama_inpainter, LAMA_AVAILABLE
except ImportError:
    LAMA_AVAILABLE = False
    
try:
    from smart_masker import SmartMasker
    _smart_masker = SmartMasker()
    SMART_MASKER_AVAILABLE = True
except ImportError:
    SMART_MASKER_AVAILABLE = False
    
from translator.context_memory import ContextMemory
from add_text import add_text

# Gemini Banana Pipeline (optional, for Gemini Full / Hybrid modes)
try:
    from gemini_pipeline import GeminiMangaPipeline
    GEMINI_PIPELINE_AVAILABLE = True
except ImportError as _gp_err:
    GEMINI_PIPELINE_AVAILABLE = False
    print(f"Gemini Pipeline not available: {_gp_err}")
from manga_ocr import MangaOcr
from ocr.chrome_lens_ocr import ChromeLensOCR
from PIL import Image
import numpy as np
import base64
import cv2


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret_key")

# Initialize SocketIO with auto-detected async mode
def get_async_mode():
    # Force threading mode: eventlet monkey-patches selectors, which breaks
    # asyncio (used by chrome-lens OCR) on Python 3.9.
    return 'threading'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=get_async_mode())

# Control verbose logging (set VERBOSE_LOG=1 to enable debug output)
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"

def log(msg):
    """Print only if verbose logging is enabled."""
    if VERBOSE_LOG:
        print(msg)

MODEL_PATH = "model/model.pt"

# Default max height for split (1.5x width = landscape-ish ratio)
DEFAULT_SPLIT_HEIGHT_RATIO = 2.0

# Global cache for OCR instances
_OCR_CACHE = {
    "chrome_lens": None,
    "manga_ocr": None
}

# Results directory for saving processed images to disk
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def cleanup_old_results(max_age_seconds=3600):
    """Remove result session directories older than max_age_seconds (default: 1 hour)."""
    try:
        cutoff = time_module.time() - max_age_seconds
        for session_dir in os.listdir(RESULTS_DIR):
            session_path = os.path.join(RESULTS_DIR, session_dir)
            if os.path.isdir(session_path) and os.path.getmtime(session_path) < cutoff:
                import shutil
                shutil.rmtree(session_path, ignore_errors=True)
    except Exception:
        pass

def split_long_image(image: np.ndarray, max_height_ratio: float = DEFAULT_SPLIT_HEIGHT_RATIO) -> list:
    """
    Split a long image into multiple shorter chunks.
    
    Args:
        image: Input image as numpy array (H, W, C)
        max_height_ratio: Maximum height/width ratio before splitting.
                          Images taller than width * ratio will be split.
                          
    Returns:
        List of image chunks (numpy arrays). If image doesn't need splitting,
        returns a list with just the original image.
    """
    height, width = image.shape[:2]
    max_height = int(width * max_height_ratio)
    
    # If image is not too tall, return as-is
    if height <= max_height:
        return [image]
    
    # Split into chunks
    chunks = []
    current_y = 0
    chunk_num = 0
    
    while current_y < height:
        # Calculate chunk end position
        chunk_end = min(current_y + max_height, height)
        
        # Extract chunk
        chunk = image[current_y:chunk_end, :].copy()
        chunks.append(chunk)
        
        current_y = chunk_end
        chunk_num += 1
    
    print(f"  Split image ({width}x{height}) into {len(chunks)} chunks")
    return chunks


@app.route("/")
def home():
    return render_template("index.html")


def process_single_image(image, manga_translator, mocr, selected_translator, selected_font, font_analyzer=None, enable_black_bubble=True):
    """Process a single image and return the translated version.
    
    Optimized with batch translation for Gemini to reduce API calls.
    Supports auto font matching when font_analyzer is provided and selected_font is 'auto'.
    """
    yolo_results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
    
    bubble_data = []
    texts_to_translate = []
    first_bubble_image = None  # For font analysis
    
    # Parse YOLO boxes
    yolo_boxes = []
    if yolo_results:
        for result in yolo_results:
            if len(result) >= 7:
                x1, y1, x2, y2, score, class_id, is_dark = result[:7]
            else:
                x1, y1, x2, y2, score, class_id = result[:6]
                is_dark = 0
            yolo_boxes.append({"coords": (int(x1), int(y1), int(x2), int(y2)), "is_dark": is_dark})

    # Hybrid Logic for Full-Page OCR
    if hasattr(mocr, 'detect_and_recognize_blocks'):
        print("Using Hybrid Detection: YOLO + Full-Page OCR blocks")
        full_blocks = mocr.detect_and_recognize_blocks(image)
        
        # Match Full-Page Blocks to YOLO Boxes
        for box in yolo_boxes:
            bx1, by1, bx2, by2 = box["coords"]
            box_texts = []
            box_trans = []
            
            # Find intersecting Full-Page blocks
            for block in list(full_blocks):
                lx1, ly1, lx2, ly2 = block["coords"]
                
                # Intersection checking
                ix1 = max(bx1, lx1)
                iy1 = max(by1, ly1)
                ix2 = min(bx2, lx2)
                iy2 = min(by2, ly2)
                
                if ix1 < ix2 and iy1 < iy2:
                    box_texts.append(block.get("text", ""))
                    if block.get("translated_text"):
                        box_trans.append(block["translated_text"])
                    full_blocks.remove(block) # Remove so it's not processed again
            
            if box_texts:
                box["text"] = " ".join([t for t in box_texts if t])
            if box_trans:
                box["translated_text"] = " ".join([t for t in box_trans if t])
        
        # Any remaining full_blocks are "outside bubbles"
        for block in full_blocks:
            yolo_boxes.append({
                "coords": block["coords"],
                "is_dark": 0,
                "text": block.get("text", ""),
                "translated_text": block.get("translated_text", ""),
                "is_outside": True
            })

    if not yolo_boxes:
        return image
        
    for box in yolo_boxes:
        x1, y1, x2, y2 = box["coords"]
        is_dark = box["is_dark"]
        is_outside = box.get("is_outside", False)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        detected_image = image[y1:y2, x1:x2]
        
        if first_bubble_image is None:
            first_bubble_image = detected_image.copy()
            
        if "text" in box:
            text = box["text"]
        else:
            im = Image.fromarray(detected_image)
            text = mocr(im)
            
        if not text or not text.strip():
            continue
            
        if is_outside:
            if LAMA_AVAILABLE:
                # Use LaMa neural inpainting for outside text
                lama = get_lama_inpainter()
                # Create text mask from the detected region using Otsu threshold
                gray_region = cv2.cvtColor(detected_image, cv2.COLOR_BGR2GRAY)
                _, text_mask = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # Dilate mask slightly to cover text edges
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                text_mask = cv2.dilate(text_mask, dilate_kernel, iterations=2)
                processed_image = lama.inpaint(detected_image, text_mask)
            else:
                # Fallback to GaussianBlur
                processed_image = cv2.GaussianBlur(detected_image, (15, 15), 0)
            cont = np.array([[[0, 0]], [[0, y2-y1]], [[x2-x1, y2-y1]], [[x2-x1, 0]]], dtype=np.int32)
            bubble_is_dark = False
            detected_color = (255, 255, 255)
            requires_stroke = True
        else:
            if SMART_MASKER_AVAILABLE:
                detected_image, cont, bubble_is_dark, detected_color = _smart_masker.clean_bubble(detected_image, force_dark=(is_dark == 1))
            else:
                detected_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
            requires_stroke = False
            
        bubble_data.append({
            'detected_image': detected_image,
            'contour': cont,
            'coords': (x1, y1, x2, y2),
            'is_dark': bubble_is_dark,
            'fill_color': detected_color,
            'requires_stroke': requires_stroke,
            'pre_translated': box.get('translated_text')
        })
        texts_to_translate.append(text)
    
    if not bubble_data:
        return image
    
    # Phase 2: Batch translate
    if selected_translator in ("gemini", "gemini-pro") and len(texts_to_translate) > 1:
        # Use batch translation for Gemini
        try:
            if manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = getattr(manga_translator, '_gemini_api_key', None)
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                gemini_model = getattr(manga_translator, '_gemini_model', 'gemini-2.5-flash')
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key, 
                    custom_prompt=custom_prompt,
                    model=gemini_model
                )
            
            translated_texts = manga_translator._gemini_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            translated_texts = [manga_translator.translate(t, method=selected_translator) for t in texts_to_translate]
    
    elif selected_translator == "copilot" and len(texts_to_translate) > 1:
        # Use batch translation for Local LLM (Ollama, LM Studio, etc.)
        try:
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                copilot_server = getattr(manga_translator, '_copilot_server', 'http://localhost:8080')
                copilot_model = getattr(manga_translator, '_copilot_model', 'gpt-4o')
                copilot_custom_prompt = getattr(manga_translator, '_copilot_custom_prompt', None)
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model}")
            
            translated_texts = manga_translator._local_llm_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            translated_texts = [manga_translator.translate(t, method=selected_translator) for t in texts_to_translate]

    elif selected_translator == "freellm" and len(texts_to_translate) > 1:
        # Use batch translation for FreeLLM
        try:
            if not hasattr(manga_translator, '_freellm_translator') or manga_translator._freellm_translator is None:
                from translator.freellm_translator import FreeLLMTranslator
                api_key = getattr(manga_translator, '_freellm_api_key', None)
                base_url = getattr(manga_translator, '_freellm_base_url', None)
                if not api_key:
                    raise ValueError("FreeLLM API key not provided")
                custom_prompt = getattr(manga_translator, '_freellm_custom_prompt', None)
                manga_translator._freellm_translator = FreeLLMTranslator(
                    api_key=api_key, 
                    base_url=base_url,
                    custom_prompt=custom_prompt
                )
            
            translated_texts = manga_translator._freellm_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            translated_texts = [manga_translator.translate(t, method=selected_translator) for t in texts_to_translate]
        except Exception as e:
            print(f"Copilot batch translation failed: {e}")
            translated_texts = texts_to_translate  # Return original on error
    
    else:
        # Single translation for other translators
        # Optimized: Use batch translation if available (e.g. for NLLB)
        translated_texts = manga_translator.translate_batch(texts_to_translate, method=selected_translator)
    
    # Phase 3: Add translated text to bubbles
    # Determine correct font path based on font name
    font_path = get_font_path(selected_font)
    for data, translated_text in zip(bubble_data, translated_texts):
        if data.get('pre_translated'):
            translated_text = data['pre_translated']
        # Use white text for dark bubbles, black text for light bubbles
        text_color = (255, 255, 255) if data.get('is_dark', False) else (0, 0, 0)
        add_text(
            image=data['detected_image'], 
            text=translated_text, 
            font_path=font_path, 
            bubble_contour=data['contour'], 
            text_color=text_color,
            is_dark_bubble=data.get('is_dark', False),
            detected_color=data.get('fill_color'),
            requires_stroke=data.get('requires_stroke', False)
        )
    
    return image


def get_font_path(font_name: str) -> str:
    """Get the correct font file path based on font name."""
    # Handle legacy fonts with 'i' suffix
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    # Yuki-* fonts use exact name
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


def process_images_with_batch(images_data, manga_translator, mocr, selected_font, translator_type, batch_size=10, use_context_memory=True, enable_black_bubble=True, ocr_engine_name="", source_lang="", target_lang="", style=""):
    """
    Process multiple images with multi-page batching for Copilot or Gemini.
    Collects all texts first, batch translates, then applies translations.
    Supports caching OCR + translation results to avoid re-processing.
    
    Args:
        images_data: List of dicts with 'image', 'name' keys
        manga_translator: MangaTranslator instance with translator
        mocr: OCR engine
        selected_font: Font to use
        translator_type: 'copilot' or 'gemini' or 'freellm'
        batch_size: Number of pages per API call
        use_context_memory: Whether to include context from all pages for better translation
        ocr_engine_name: OCR engine name for cache key
        source_lang: Source language for cache key
        target_lang: Target language for cache key
        style: Translation style for cache key
        
    Returns:
        List of processed images with translations applied
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from translator.translation_cache import get_cache
    
    cache = get_cache()
    
    def emit_progress(phase, current, total, message):
        """Emit progress update via WebSocket."""
        try:
            socketio.emit('progress', {
                'phase': phase,
                'current': current,
                'total': total,
                'message': message,
                'percent': int((current / max(total, 1)) * 100)
            })
        except Exception as e:
            pass  # Silently fail if socket not connected
    
    total_images = len(images_data)
    log(f"Processing {total_images} images... Context Memory: {'ON' if use_context_memory else 'OFF'}")
    
    start_time = time.time()
    
    # Check if using Chrome Lens OCR (has batch support)
    use_batch_ocr = hasattr(mocr, 'process_batch')
    
    # Pre-check cache for all images
    cached_pages = {}  # {page_name: cached_data}
    cache_hits = 0
    for img_data in images_data:
        image = img_data['image']
        name = img_data['name']
        _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = img_encoded.tobytes()
        cached = cache.get(image_bytes, ocr_engine_name, source_lang, translator_type, target_lang, style)
        if cached:
            cached_pages[name] = cached
            cache_hits += 1
    
    if cache_hits > 0:
        print(f"📦 Cache: {cache_hits}/{total_images} pages found in cache (skipping OCR + translation)")
        emit_progress('cache', cache_hits, total_images, f'Tìm thấy {cache_hits}/{total_images} trang trong cache')

    # Phase 1a: Detect bubbles and collect all bubble images
    print("\n[Phase 1] Detecting bubbles...")
    emit_progress('detection', 0, total_images, 'Bắt đầu phát hiện speech bubbles...')
    all_pages_data = {}  # {page_name: {'image': img, 'bubbles': [...], 'bubble_images': [...]}}
    all_bubble_images = []  # Flat list for batch OCR
    bubble_mapping = []  # [(page_name, bubble_idx), ...] to map back
    
    for idx, img_data in enumerate(images_data):
        image = img_data['image']
        name = img_data['name']
        
        emit_progress('detection', idx + 1, total_images, f'Phát hiện bubbles: {name}')
        print(f"  [{idx+1}/{total_images}] {name}", end="", flush=True)
        
        bubble_data = []
        page_texts = []
        cleaned_image = None
        
        if PCLEANER_AVAILABLE:
            # === PanelCleanerZ Pipeline ===
            # Step 1: Detect text blocks + generate pixel-level mask + clean image
            result = _pcleaner.detect_and_clean(image)
            cleaned_image = result['cleaned_image']
            ctd_blocks = result['text_blocks']
            mask_refined = result['mask_refined']
            
            # NOTE: Do NOT replace image here - OCR needs the original text!
            # cleaned_image will be stored and applied in Phase 4 before rendering.
            
            # --- OUTSIDE-BUBBLE TEXT DETECTION (Full-Page OCR) ---
            if hasattr(mocr, 'detect_and_recognize_blocks'):
                print("Using Full-Page OCR blocks in PanelCleanerZ pipeline")
                full_blocks = mocr.detect_and_recognize_blocks(image)
                
                # Match Full-Page Blocks to CTD Blocks
                for box in ctd_blocks:
                    bx1, by1, bx2, by2 = box["coords"]
                    box_texts = []
                    box_trans = []
                    
                    for block in list(full_blocks):
                        lx1, ly1, lx2, ly2 = block["coords"]
                        ix1 = max(bx1, lx1)
                        iy1 = max(by1, ly1)
                        ix2 = min(bx2, lx2)
                        iy2 = min(by2, ly2)
                        
                        if ix1 < ix2 and iy1 < iy2:
                            box_texts.append(block.get("text", ""))
                            if block.get("translated_text"):
                                box_trans.append(block["translated_text"])
                            full_blocks.remove(block)
                            
                    if box_texts:
                        box["pre_ocr_text"] = " ".join([t for t in box_texts if t])
                    if box_trans:
                        box["translated_text"] = " ".join([t for t in box_trans if t])
                        
                # Add remaining full_blocks as new outside bubbles
                for block in full_blocks:
                    ctd_blocks.append({
                        "coords": block["coords"],
                        "pre_ocr_text": block.get("text", ""),
                        "translated_text": block.get("translated_text", ""),
                        "from_lens": True,
                        "bg_color": (255, 255, 255)
                    })
            # --------------------------------------
            
            print(f" - CTD found {len(ctd_blocks)} text blocks", end="", flush=True)
            
            # Step 2: For each text block, OCR from original image
            for blk in ctd_blocks:
                x1, y1, x2, y2 = blk['coords']
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                detected_region = image[y1:y2, x1:x2]
                
                if blk.get("from_lens"):
                    bubble_mapping.append((name, len(page_texts)))
                    page_texts.append(blk["pre_ocr_text"])
                    processed_image = cv2.GaussianBlur(detected_region, (15, 15), 0)
                    cont = np.array([[[0, 0]], [[0, y2-y1]], [[x2-x1, y2-y1]], [[x2-x1, 0]]], dtype=np.int32)
                    bubble_data.append({
                        'detected_image': processed_image,
                        'contour': cont,
                        'coords': (x1, y1, x2, y2),
                        'is_dark': False,
                        'fill_color': (255, 255, 255),
                        'requires_stroke': True,
                        'pre_translated': blk.get('translated_text')
                    })
                    continue
                
                cleaned_region = cleaned_image[y1:y2, x1:x2]
                
                # OCR on the original (uncleaned) image region
                if "pre_ocr_text" in blk:
                    bubble_mapping.append((name, len(page_texts)))
                    page_texts.append(blk["pre_ocr_text"])
                else:
                    im = Image.fromarray(detected_region)
                    all_bubble_images.append(im)
                    bubble_mapping.append((name, len(page_texts)))
                    page_texts.append(None)  # Placeholder
                
                # Determine if dark bubble from CTD colors
                bg_r, bg_g, bg_b = blk['bg_color']
                avg_bg = (bg_r + bg_g + bg_b) / 3
                bubble_is_dark = avg_bg < 128
                
                # Use the cleaned region directly
                detected_color = (int(bg_b), int(bg_g), int(bg_r))  # RGB -> BGR
                cont = np.array([[[0, 0]], [[0, y2-y1]], [[x2-x1, y2-y1]], [[x2-x1, 0]]], dtype=np.int32)
                
                # Check if outside bubble (complex background)
                _, is_uniform = _pcleaner._analyze_block_background(
                    image[y1:y2, x1:x2],
                    mask_refined[y1:y2, x1:x2] if mask_refined is not None else np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                )
                requires_stroke = not is_uniform
                
                bubble_data.append({
                    'detected_image': cleaned_region.copy(),
                    'contour': cont,
                    'coords': (x1, y1, x2, y2),
                    'is_dark': bubble_is_dark,
                    'fill_color': detected_color,
                    'requires_stroke': requires_stroke,
                    'pre_translated': blk.get('translated_text')
                })
            
            print(f" ✓")
        else:
            # === Fallback: Original YOLO Pipeline ===
            yolo_results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
            yolo_boxes = []
            if yolo_results:
                for result in yolo_results:
                    if len(result) >= 7:
                        x1, y1, x2, y2, score, class_id, is_dark = result[:7]
                    else:
                        x1, y1, x2, y2, score, class_id = result[:6]
                        is_dark = 0
                    yolo_boxes.append({"coords": (int(x1), int(y1), int(x2), int(y2)), "is_dark": is_dark})
                    
            # Hybrid Logic for Full-Page OCR
            if hasattr(mocr, 'detect_and_recognize_blocks'):
                full_blocks = mocr.detect_and_recognize_blocks(image)
                for box in yolo_boxes:
                    bx1, by1, bx2, by2 = box["coords"]
                    box_texts = []
                    box_trans = []
                    for block in list(full_blocks):
                        lx1, ly1, lx2, ly2 = block["coords"]
                        ix1 = max(bx1, lx1)
                        iy1 = max(by1, ly1)
                        ix2 = min(bx2, lx2)
                        iy2 = min(by2, ly2)
                        if ix1 < ix2 and iy1 < iy2:
                            box_texts.append(block.get("text", ""))
                            if block.get("translated_text"):
                                box_trans.append(block["translated_text"])
                            full_blocks.remove(block)
                    if box_texts:
                        box["text"] = " ".join([t for t in box_texts if t])
                    if box_trans:
                        box["translated_text"] = " ".join([t for t in box_trans if t])
                for block in full_blocks:
                    yolo_boxes.append({
                        "coords": block["coords"],
                        "is_dark": 0,
                        "text": block.get("text", ""),
                        "translated_text": block.get("translated_text", ""),
                        "is_outside": True
                    })

            if not yolo_boxes:
                all_pages_data[name] = {'image': image, 'bubbles': [], 'texts': []}
                print(f" - 0 bubbles")
                continue
            
            print(f" - {len(yolo_boxes)} bubbles")
            
            for bubble_idx, box in enumerate(yolo_boxes):
                x1, y1, x2, y2 = box["coords"]
                is_dark = box["is_dark"]
                is_outside = box.get("is_outside", False)
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                detected_image = image[y1:y2, x1:x2]
                
                if "text" in box:
                    text = box["text"]
                    if not text or not text.strip():
                        continue
                    page_texts.append(text)
                    if is_outside:
                        processed_image = cv2.GaussianBlur(detected_image, (15, 15), 0)
                        cont = np.array([[[0, 0]], [[0, y2-y1]], [[x2-x1, y2-y1]], [[x2-x1, 0]]], dtype=np.int32)
                        bubble_is_dark = False
                        detected_color = (255, 255, 255)
                        requires_stroke = True
                    else:
                        processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
                        requires_stroke = False
                    bubble_data.append({
                        'detected_image': processed_image,
                        'contour': cont,
                        'coords': (x1, y1, x2, y2),
                        'is_dark': bubble_is_dark,
                        'fill_color': detected_color,
                        'requires_stroke': requires_stroke,
                        'pre_translated': box.get('translated_text')
                    })
                else:
                    all_bubble_images.append(Image.fromarray(detected_image.copy()))
                    bubble_mapping.append((name, len(page_texts)))
                    page_texts.append(None)
                    processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
                    bubble_data.append({
                        'detected_image': processed_image,
                        'contour': cont,
                        'coords': (x1, y1, x2, y2),
                        'is_dark': bubble_is_dark,
                        'fill_color': detected_color,
                        'requires_stroke': False,
                        'pre_translated': box.get('translated_text')
                    })
        
        all_pages_data[name] = {
            'image': image,
            'cleaned_image': cleaned_image,
            'bubbles': bubble_data,
            'texts': page_texts
        }

    detection_time = time.time() - start_time
    print(f"✓ Bubble detection completed in {detection_time:.1f}s ({len(all_bubble_images)} total bubbles)")
    emit_progress('detection', total_images, total_images, f'Phát hiện xong {len(all_bubble_images)} bubbles')
    
    # Phase 1b: Batch OCR all bubbles at once
    if all_bubble_images:
        ocr_start = time.time()
        emit_progress('ocr', 0, 1, f'Đang OCR {len(all_bubble_images)} bubbles...')
        print(f"\n[Phase 2] OCR processing {len(all_bubble_images)} bubbles...", end=" ", flush=True)
        
        if use_batch_ocr:
            # Use concurrent batch OCR (Chrome Lens)
            all_texts = mocr.process_batch(all_bubble_images)
        else:
            # Sequential OCR (MangaOcr or others)
            all_texts = [mocr(img) for img in all_bubble_images]
        
        # Now map the texts back to the bubbles preserving order
        for (page_name, text_idx), text in zip(bubble_mapping, all_texts):
            all_pages_data[page_name]['texts'][text_idx] = text
            
        # Clean up any None values (if any OCR failed) to preserve length matching bubbles
        for page_name in all_pages_data:
            all_pages_data[page_name]['texts'] = [t if t is not None else "" for t in all_pages_data[page_name]['texts']]
        
        ocr_time = time.time() - ocr_start
        print(f"({ocr_time:.1f}s)")
        print(f"✓ OCR completed in {ocr_time:.1f}s ({len(all_bubble_images)/ocr_time:.1f} bubbles/sec)")
        emit_progress('ocr', 1, 1, f'OCR hoàn tất ({len(all_bubble_images)} bubbles)')
    
    # Phase 3: Batch translate all pages together
    emit_progress('translation', 0, 1, 'Đang dịch...')
    
    # Separate cached vs uncached pages
    all_translations = {}
    uncached_pages_texts = {}
    
    for name, data in all_pages_data.items():
        if name in cached_pages and data['texts']:
            # Use cached translations
            cached = cached_pages[name]
            cached_ocr = cached.get('ocr_texts', [])
            cached_trans = cached.get('translated_texts', [])
            
            # Verify cache matches current bubble count
            if len(cached_trans) == len(data['texts']):
                all_translations[name] = cached_trans
                # Also replace OCR texts with cached ones for logging
                data['texts'] = cached_ocr if len(cached_ocr) == len(data['texts']) else data['texts']
                print(f"  [✓ CACHED] {name}: {len(cached_trans)} translations")
            else:
                # Cache mismatch (different bubble count), need to re-translate
                print(f"  [✗ CACHE MISMATCH] {name}: cached={len(cached_trans)}, current={len(data['texts'])}")
                if data['texts']:
                    uncached_pages_texts[name] = data['texts']
        elif data['texts']:
            uncached_pages_texts[name] = data['texts']
    
    if uncached_pages_texts:
        # Get the translator based on type
        if translator_type == "copilot" and hasattr(manga_translator, '_local_llm_translator') and manga_translator._local_llm_translator:
            translator = manga_translator._local_llm_translator
            translator_name = "Local LLM"
        elif translator_type in ("gemini", "gemini-pro") and hasattr(manga_translator, '_gemini_translator') and manga_translator._gemini_translator:
            translator = manga_translator._gemini_translator
            translator_name = "Gemini Pro" if translator_type == "gemini-pro" else "Gemini Flash"
        elif translator_type == "freellm" and hasattr(manga_translator, '_freellm_translator') and manga_translator._freellm_translator:
            translator = manga_translator._freellm_translator
            translator_name = "FreeLLM"
        else:
            translator = None
            translator_name = "Unknown"
        
        if translator:
            cached_count = len(all_translations)
            total_count = cached_count + len(uncached_pages_texts)
            print(f"{translator_name} batch translating {len(uncached_pages_texts)} pages in chunks of {batch_size}... ({cached_count} cached, {len(uncached_pages_texts)} new)")
            
            # Initialize context memory if enabled
            context_memory = None
            if use_context_memory:
                context_memory = ContextMemory()
                print(f"  Context Memory enabled - tracking terms and story context")
            
            # Process in batches
            page_names = list(uncached_pages_texts.keys())
            
            for i in range(0, len(page_names), batch_size):
                batch_names = page_names[i:i + batch_size]
                batch_texts = {name: uncached_pages_texts[name] for name in batch_names}
                
                print(f"  Translating batch {i//batch_size + 1}: pages {i+1}-{min(i+batch_size, len(page_names))}")
                
                try:
                    translated = translator.translate_pages_batch(
                        batch_texts,
                        source=manga_translator.source,
                        target=manga_translator.target,
                        context_memory=context_memory
                    )
                    all_translations.update(translated)
                    
                    # Save new translations to cache
                    for page_name in batch_names:
                        if page_name in translated:
                            # Find original image for cache key
                            for img_data in images_data:
                                if img_data['name'] == page_name:
                                    _, img_enc = cv2.imencode('.jpg', img_data['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    cache.put(
                                        img_enc.tobytes(), ocr_engine_name, source_lang,
                                        translator_type, target_lang, style,
                                        {
                                            'ocr_texts': all_pages_data[page_name]['texts'],
                                            'translated_texts': translated[page_name],
                                            'timestamp': time.time()
                                        }
                                    )
                                    break
                    
                    # Update context memory with this batch's translations
                    if context_memory:
                        context_memory.update_from_translation(batch_texts, translated)
                        stats = context_memory.get_stats()
                        print(f"    Context updated: {stats['tracked_words']} terms tracked, {stats['recent_pages']} pages in memory")
                        
                except Exception as e:
                    print(f"  Batch failed: {e}, falling back to individual translation")
                    for name, texts in batch_texts.items():
                        try:
                            all_translations[name] = translator.translate_batch(
                                texts, manga_translator.source, manga_translator.target
                            )
                            time.sleep(2)  # Delay between individual translations
                        except:
                            all_translations[name] = texts  # Return original on error
                
                # Delay between batches to avoid rate limiting
                if i + batch_size < len(page_names):
                    time.sleep(3)
                    print(f"    (3s delay between batches to avoid rate limit)")
    
    translation_time = time.time() - start_time - detection_time
    print(f"✓ Translation completed in {translation_time:.1f}s")
    emit_progress('translation', 1, 1, 'Dịch hoàn tất')
    
    # Phase 4: Apply translations and render text
    emit_progress('rendering', 0, total_images, 'Đang render text vào ảnh...')
    render_start = time.time()
    processed_results = []
    font_path = get_font_path(selected_font)
    
    print(f"\n[Phase 4] Rendering text...")
    
    render_idx = 0
    for name, data in all_pages_data.items():
        render_idx += 1
        emit_progress('rendering', render_idx, total_images, f'Render text: {name}')
        
        image = data['image']
        bubbles = data['bubbles']
        translated_texts = all_translations.get(name, data['texts'])  # Fallback to original
        
        print(f"  [{name}] {len(bubbles)} bubbles, font={font_path}")
        
        # Log full text: original OCR vs translated
        original_texts = data['texts']
        for i, (orig, trans) in enumerate(zip(original_texts, translated_texts)):
            print(f"    [{i+1}] OCR: {orig}")
            print(f"         -> : {trans}")
        
        # Apply cleaned image (text erased) before rendering translated text
        if data.get('cleaned_image') is not None:
            image[:] = data['cleaned_image']
        
        # Apply text to bubbles on the CLEANED image
        for bubble, text in zip(bubbles, translated_texts):
            if bubble.get('pre_translated'):
                text = bubble['pre_translated']
            x1, y1, x2, y2 = bubble['coords']
            # Get the region in the original image (this is a view, modifications affect original)
            bubble_region = image[y1:y2, x1:x2]
            # Use white text for dark bubbles, black text for light bubbles
            text_color = (255, 255, 255) if bubble.get('is_dark', False) else (0, 0, 0)
            # Add translated text
            add_text(
                image=bubble_region, 
                text=text, 
                font_path=font_path, 
                bubble_contour=bubble['contour'], 
                text_color=text_color,
                is_dark_bubble=bubble.get('is_dark', False),
                detected_color=bubble.get('fill_color'),
                requires_stroke=bubble.get('requires_stroke', False)
            )
        
        processed_results.append({
            'image': image,
            'name': name
        })
    
    render_time = time.time() - render_start
    total_time = time.time() - start_time
    
    print(f"✓ Text rendering completed in {render_time:.1f}s")
    print(f"{'='*50}")
    print(f"✓ TOTAL: {total_images} images processed in {total_time:.1f}s ({total_time/total_images:.1f}s/image)")
    print(f"{'='*50}\n")
    
    emit_progress('done', total_images, total_images, f'Hoàn tất! {total_images} ảnh trong {total_time:.1f}s')
    
    return processed_results


_pending_jobs = {}
_active_jobs = {}  # session_id -> {'status': 'running'|'completed'|'failed', 'total': N, 'error': ''}
@socketio.on('start_translation')
def handle_start_translation(data):
    """Client triggers this after uploading files via POST."""
    session_id = data.get('session_id')
    if session_id and session_id in _pending_jobs:
        job = _pending_jobs.pop(session_id)
        socketio.start_background_task(run_translation_job, job, request.sid)

def run_translation_job(job, sid=None):
    import hashlib
    session_id = job['session_id']
    session_dir = job['session_dir']
    saved_files = job['saved_files']
    source_lang = job['source_lang']
    target_lang = job['target_lang']
    font_path = job['font_path']
    split_long_images = job['split_long_images']
    upscale_image = job['upscale_image']
    gemini_api_key = job['gemini_api_key']
    
    total = len(saved_files)
    _active_jobs[session_id] = {'status': 'running', 'total': total, 'completed': 0, 'error': ''}
    
    # --- Translation cache ---
    CACHE_DIR = os.path.join(RESULTS_DIR, '_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    def get_cache_key(img_path):
        """Generate cache key from image content + translation settings."""
        with open(img_path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        settings = f"{source_lang}_{target_lang}_{upscale_image}_{font_path}"
        settings_hash = hashlib.md5(settings.encode()).hexdigest()[:8]
        return f"{img_hash}_{settings_hash}"
    
    def emit_gemini_progress(current, total_count, message):
        try:
            socketio.emit('progress', {
                'phase': 'gemini',
                'current': current,
                'total': total_count,
                'message': message,
                'percent': int((current / max(total_count, 1)) * 100)
            })
        except Exception:
            pass

    emit_gemini_progress(0, total, '🍌 Bắt đầu Gemini Pipeline (hybrid)...')
    
    try:
        pipeline = GeminiMangaPipeline(
            api_key=gemini_api_key or None,
            strategy="hybrid",
            max_workers=3,
            temperature=0.1
        )
    except Exception as e:
        print(f"⚠️ Gemini Pipeline init failed: {e}")
        _active_jobs[session_id]['status'] = 'failed'
        _active_jobs[session_id]['error'] = str(e)
        return

    print(f"Processing {total} pages with Gemini Hybrid...")
    cached_count = 0

    for i, img_data in enumerate(saved_files):
        try:
            base_name = img_data['name']
            
            # --- Check translation cache ---
            cache_key = get_cache_key(img_data['path'])
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.jpg")
            
            if os.path.exists(cache_path):
                # Cache hit! Copy cached result to session dir and emit
                cached_count += 1
                import shutil
                filename = f"{base_name}.jpg"
                filepath = os.path.join(session_dir, filename)
                shutil.copy2(cache_path, filepath)
                
                print(f"  ✅ Page {i+1} '{base_name}': cache hit → skipped")
                emit_gemini_progress(i + 1, total, f'Trang {i+1}/{total}: Cache ✓')
                
                socketio.emit('page_result', {
                    'name': base_name,
                    'url': f'/static/results/{session_id}/{filename}',
                    'page_num': i + 1,
                    'total': total
                })
                _active_jobs[session_id]['completed'] = i + 1
                continue
            
            image = cv2.imread(img_data['path'])
            original = cv2.imread(img_data['original_path'])
            
            # --- Optional: Upscale blurry images ---
            print(f"  [DEBUG] upscale_image={upscale_image}, UPSCALER_AVAILABLE={UPSCALER_AVAILABLE}")
            if upscale_image and UPSCALER_AVAILABLE:
                emit_gemini_progress(i, total, f'Trang {i+1}/{total}: Upscale...')
                try:
                    upscaler = get_manga_upscaler(scale=4)
                    image = upscaler.upscale(image)
                    original = upscaler.upscale(original)
                    print(f"  ✓ Upscaled to {image.shape[1]}x{image.shape[0]}")
                except Exception as e:
                    print(f"  ⚠️ Upscale failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            img_h, img_w = image.shape[:2]
            
            emit_gemini_progress(i, total, f'Trang {i+1}/{total}: Phân tích...')
            
            # --- Phase 1: PanelCleanerZ detect + clean ---
            detected_blocks = []
            pcleaner_cleaned = False
            
            if PCLEANER_AVAILABLE:
                try:
                    result = _pcleaner.detect_and_clean(image)
                    image = result['cleaned_image']
                    detected_blocks = result.get('text_blocks', [])
                    pcleaner_cleaned = True
                    print(f"\n  Page {i+1}: PanelCleanerZ found {len(detected_blocks)} text regions")
                    for j, blk in enumerate(detected_blocks):
                        x1, y1, x2, y2 = blk['coords']
                        vert = '↕' if blk.get('vertical') else '↔'
                        lang = blk.get('language', '?')
                        print(f"    [{j+1}] {vert} {lang} ({x1},{y1})-({x2},{y2}) {x2-x1}x{y2-y1}")
                except Exception as e:
                    print(f"  ⚠️ PanelCleanerZ failed: {e}")
            
            if not detected_blocks:
                # Fallback: use Gemini analysis if PanelCleanerZ unavailable
                print(f"  Fallback: using Gemini for detection...")
                analysis = pipeline.analyze_page(original, source_lang, target_lang)
                if analysis:
                    # Convert Gemini format to PanelCleanerZ format + render directly
                    for block in analysis:
                        coords = block.get('coords', [])
                        if len(coords) == 4:
                            x1, y1, x2, y2 = [int(c) for c in coords]
                            text = block.get('translated_text', '').strip()
                            if text and x2 > x1 and y2 > y1:
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(img_w, x2), min(img_h, y2)
                                region = image[y1:y2, x1:x2]
                                avg = np.mean(region)
                                fill_color = (255,255,255) if avg > 128 else (0,0,0)
                                text_color = (0,0,0) if avg > 128 else (255,255,255)
                                cv2.rectangle(image, (x1,y1), (x2,y2), fill_color, -1)
                                region = image[y1:y2, x1:x2]
                                cont = np.array([[[0,0]], [[0,y2-y1]], [[x2-x1,y2-y1]], [[x2-x1,0]]], dtype=np.int32)
                                add_text(image=region, text=text, font_path=font_path, 
                                        bubble_contour=cont, text_color=text_color,
                                        is_dark_bubble=(avg < 128), requires_stroke=True)
                    print(f"  Gemini fallback rendered {len(analysis)} blocks")
                
            elif detected_blocks:
                # --- Phase 2: Gemini batch translate ---
                emit_gemini_progress(i, total, f'Trang {i+1}/{total}: Dịch thuật...')
                
                source_name = {'ja': 'Japanese', 'zh': 'Chinese', 'ko': 'Korean', 'en': 'English'}.get(source_lang, source_lang)
                target_name = {'vi': 'Vietnamese', 'en': 'English'}.get(target_lang, target_lang)
                
                # Draw numbered boxes on image copy so Gemini can identify regions
                annotated = original.copy()
                for j, blk in enumerate(detected_blocks):
                    x1, y1, x2, y2 = blk['coords']
                    color = (0, 0, 255)  # Red in BGR
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = str(j + 1)
                    cv2.putText(annotated, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # --- Phase 2a: Two-pass OCR for complex vertical regions ---
                import time
                pre_ocr_texts = {}  # region_id -> ocr'd text
                COMPLEX_WIDTH_THRESHOLD = 300  # pixels at 4x - roughly 3+ columns
                
                complex_regions = []
                for j, blk in enumerate(detected_blocks):
                    x1, y1, x2, y2 = blk['coords']
                    is_vert = blk.get('vertical', False)
                    bw = x2 - x1
                    if is_vert and bw > COMPLEX_WIDTH_THRESHOLD:
                        complex_regions.append((j, blk))
                
                if complex_regions:
                    from google.genai import types as ocr_types
                    print(f"  🔍 Pre-OCR: {len(complex_regions)} complex vertical regions")
                    
                    for idx, (j, blk) in enumerate(complex_regions):
                        x1, y1, x2, y2 = blk['coords']
                        cx1, cy1 = max(0, x1), max(0, y1)
                        cx2, cy2 = min(img_w, x2), min(img_h, y2)
                        crop = original[cy1:cy2, cx1:cx2]
                        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        
                        ocr_prompt = f"""Read ALL the {source_name} text in this image. 
The text is written VERTICALLY in columns. Read columns from RIGHT to LEFT, each column from TOP to BOTTOM.

IMPORTANT: Read each column COMPLETELY before moving to the next column to the left.

Return ONLY the raw text, nothing else. No translation, no explanation."""
                        
                        # Retry with exponential backoff for rate limits
                        for ocr_attempt in range(3):
                            try:
                                ocr_response = pipeline.client.models.generate_content(
                                    model="gemini-2.5-pro",
                                    contents=[pil_crop, ocr_prompt],
                                    config=ocr_types.GenerateContentConfig(
                                        temperature=0.1
                                    )
                                )
                                ocr_text = ocr_response.text.strip()
                                if ocr_text:
                                    pre_ocr_texts[j + 1] = ocr_text
                                    print(f"    Region {j+1}: OCR → {ocr_text[:60]}...")
                                break
                            except Exception as ocr_e:
                                err_str = str(ocr_e).lower()
                                is_retryable = any(k in err_str for k in ['429', 'connection', 'reset', 'timeout', 'readerror', 'unavailable', '503', '500'])
                                if is_retryable and ocr_attempt < 2:
                                    wait = (ocr_attempt + 1) * 15
                                    print(f"    Region {j+1}: Retryable error, waiting {wait}s... ({ocr_e})")
                                    time.sleep(wait)
                                else:
                                    print(f"    Region {j+1}: OCR failed → {ocr_e}")
                                    break
                        
                        # Delay between requests to avoid rate limits
                        if idx < len(complex_regions) - 1:
                            time.sleep(3)
                
                # Build per-region direction hints
                region_hints = []
                for j, blk in enumerate(detected_blocks):
                    x1, y1, x2, y2 = blk['coords']
                    is_vert = blk.get('vertical', False)
                    lang = blk.get('language', 'unknown')
                    direction = "VERTICAL (read top→bottom, columns right→left)" if is_vert else "HORIZONTAL (left→right)"
                    region_hints.append(f"  Region {j+1}: ({x1},{y1})-({x2},{y2}) {direction} [{lang}]")
                region_hints_str = "\n".join(region_hints)
                
                # Build pre-OCR section if we have any
                pre_ocr_section = ""
                if pre_ocr_texts:
                    pre_ocr_lines = []
                    for rid, text in sorted(pre_ocr_texts.items()):
                        pre_ocr_lines.append(f"  Region {rid}: {text}")
                    pre_ocr_section = f"""

=== PRE-OCR'D TEXT (USE THIS AS GROUND TRUTH) ===
The following regions have been pre-OCR'd with high accuracy. Use this text as the source for translation.
Do NOT re-read these regions from the image. Trust the pre-OCR'd text below:
{chr(10).join(pre_ocr_lines)}
=== END PRE-OCR'D TEXT ==="""
                
                translate_prompt = f"""This manga page has {len(detected_blocks)} text regions marked with red numbered rectangles.

For each numbered region (1 to {len(detected_blocks)}), read the {source_name} text INSIDE that specific red box and translate to {target_name}.

TEXT DIRECTION for each region:
{region_hints_str}
{pre_ocr_section}

=== CRITICAL: HOW TO READ VERTICAL CJK TEXT ===
For regions marked VERTICAL (Chinese/Japanese), text is arranged in COLUMNS running top-to-bottom.
If a region contains MULTIPLE COLUMNS, you MUST follow these steps:

Step 1: Identify how many vertical columns of text exist in the region.
Step 2: Start with the RIGHTMOST column. Read ALL characters in that column from TOP to BOTTOM.
Step 3: Move to the NEXT column to the LEFT. Read ALL its characters from TOP to BOTTOM.
Step 4: Repeat until all columns are read.

CRITICAL: Read each column COMPLETELY before moving to the next column.
NEVER interleave or mix characters from different columns!

Example - A region with 3 vertical columns:
  [Col3] [Col2] [Col1]    ← Col1 is rightmost
   因      卑      啓
   為      職      稟
   這      有      公
   名      一      公
   盜      個
   賊      發
          現

  ✓ CORRECT reading: 啓稟公公 → 卑職有一個發現 → 因為這名盜賊
  ✗ WRONG reading:   啓卑因稟職為公有這公一名 (interleaving columns!)

For HORIZONTAL regions: read left to right, top to bottom (normal reading order).
=== END READING RULES ===

Return a JSON array with {len(detected_blocks)} objects, one per numbered box, in ORDER.
Each object must have: "region_id" (integer), "original_text" (string), "translated_text" (string).

IMPORTANT: Match each region_id to the number shown on the image. Read the text INSIDE each red box carefully.

Translation rules:
- Natural spoken {target_name}, NOT textbook style
- Keep character names unchanged
- Dialog should sound natural when read aloud
- OCR text may have errors - use context to auto-correct before translating"""

                if target_lang == "vi":
                    translate_prompt += """
- Use proper Vietnamese pronouns (tao/mày for rough, tôi/anh for polite)
- Historical terms: Sino-Vietnamese (皇帝→Hoàng đế, 公公→Công Công)
- SFX: translate naturally"""

                translate_prompt += "\n\nReturn ONLY the JSON array."
                
                try:
                    from google.genai import types
                    # Send ANNOTATED image (with numbered boxes)
                    pil_annotated = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    
                    translations = None
                    for attempt in range(5):
                        try:
                            response = pipeline.client.models.generate_content(
                                model="gemini-2.5-pro",
                                contents=[pil_annotated, translate_prompt],
                                config=types.GenerateContentConfig(
                                    temperature=0.1,
                                    response_mime_type="application/json"
                                )
                            )
                        except Exception as api_e:
                            err_str = str(api_e).lower()
                            is_retryable = any(k in err_str for k in ['429', 'connection', 'reset', 'timeout', 'readerror', 'unavailable', '503', '500'])
                            if is_retryable and attempt < 4:
                                wait = (attempt + 1) * 15
                                print(f"  ⚠️ Retryable error (attempt {attempt+1}/5), waiting {wait}s... ({api_e})")
                                time.sleep(wait)
                                continue
                            else:
                                raise
                        
                        try:
                            translations = json.loads(response.text)
                            break
                        except json.JSONDecodeError as je:
                            if attempt < 4:
                                print(f"  ⚠️ JSON parse error (attempt {attempt+1}/5), retrying...")
                            else:
                                raise je
                    
                    print(f"  ✓ Gemini translated {len(translations)} regions")
                    
                    # Map translations to detected blocks
                    trans_map = {}
                    for t in translations:
                        rid = t.get('region_id', 0)
                        orig = t.get('original_text', '')
                        trans = t.get('translated_text', '')
                        trans_map[rid] = trans
                        print(f"    Region {rid}:")
                        print(f"      原文: {orig}")
                        print(f"      Dịch: {trans}")
                    
                except Exception as e:
                    print(f"  ⚠️ Gemini translation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    trans_map = {}
                
                # --- Phase 3: Render translations ---
                emit_gemini_progress(i, total, f'Trang {i+1}/{total}: Chèn chữ...')
                rendered_count = 0
                
                # Collect all block coords for overlap checking
                all_coords = [blk['coords'] for blk in detected_blocks]
                
                for j, blk in enumerate(detected_blocks):
                    try:
                        text = trans_map.get(j + 1, '').strip()
                        if not text:
                            continue
                        
                        x1, y1, x2, y2 = blk['coords']
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_w, x2), min(img_h, y2)
                        bw, bh = x2 - x1, y2 - y1
                        
                        if bw <= 0 or bh <= 0:
                            continue
                        
                        # --- Expand narrow vertical boxes for horizontal text ---
                        is_vert = blk.get('vertical', False)
                        min_width = 100  # Minimum width for readable horizontal text
                        
                        if bw < min_width and bh > bw * 1.5:
                            # Need to expand width
                            needed = min_width - bw
                            # Try expanding both sides equally
                            expand_left = needed // 2
                            expand_right = needed - expand_left
                            
                            new_x1 = max(0, x1 - expand_left)
                            new_x2 = min(img_w, x2 + expand_right)
                            
                            # Check overlap with OTHER blocks
                            overlap = False
                            for k, other_coords in enumerate(all_coords):
                                if k == j:
                                    continue
                                ox1, oy1, ox2, oy2 = other_coords
                                # Check if expanded box overlaps other block
                                if new_x1 < ox2 and new_x2 > ox1 and y1 < oy2 and y2 > oy1:
                                    overlap = True
                                    break
                            
                            if not overlap:
                                # Fill expanded area with background
                                bg_color = blk.get('bg_color', (255, 255, 255))
                                # Fill left expansion
                                if new_x1 < x1:
                                    cv2.rectangle(image, (new_x1, y1), (x1, y2), bg_color, -1)
                                # Fill right expansion
                                if new_x2 > x2:
                                    cv2.rectangle(image, (x2, y1), (new_x2, y2), bg_color, -1)
                                
                                x1, x2 = new_x1, new_x2
                                bw = x2 - x1
                                print(f"    ↔ Expanded to {bw}x{bh}")
                        
                        # Use PanelCleanerZ's detected colors
                        bg_color = blk.get('bg_color', (255, 255, 255))
                        bg_brightness = sum(bg_color) / 3
                        is_dark = bg_brightness < 128
                        text_color = (255, 255, 255) if is_dark else (0, 0, 0)
                        
                        # Get cleaned region
                        region = image[y1:y2, x1:x2]
                        
                        # Create contour for add_text
                        cont = np.array([
                            [[0, 0]], [[0, bh]], [[bw, bh]], [[bw, 0]]
                        ], dtype=np.int32)
                        
                        # Render with stroke for readability
                        add_text(
                            image=region,
                            text=text,
                            font_path=font_path,
                            bubble_contour=cont,
                            text_color=text_color,
                            is_dark_bubble=is_dark,
                            requires_stroke=True
                        )
                        
                        rendered_count += 1
                        vert = '↕' if blk.get('vertical') else '↔'
                        print(f"    ✓ [{vert}] ({x1},{y1})-({x2},{y2}) {bw}x{bh}: {text[:35]}...")
                        
                    except Exception as e:
                        print(f"    ⚠️ Block {j+1} render error: {e}")
                
                print(f"  ✓ Rendered {rendered_count}/{len(detected_blocks)} blocks")
            
            # Save result
            if split_long_images:
                chunks = split_long_image(image)
            else:
                chunks = [image]
            
            for j, chunk in enumerate(chunks):
                chunk_name = f"{base_name}_part{j+1}" if len(chunks) > 1 else base_name
                filename = f"{chunk_name}.jpg"
                filepath = os.path.join(session_dir, filename)
                cv2.imwrite(filepath, chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save to persistent cache (only for single-chunk pages)
                if len(chunks) == 1:
                    try:
                        import shutil
                        shutil.copy2(filepath, cache_path)
                    except Exception:
                        pass
                
                socketio.emit('page_result', {
                    'name': chunk_name,
                    'url': f'/static/results/{session_id}/{filename}',
                    'page_num': i + 1,
                    'total': total
                })
            _active_jobs[session_id]['completed'] = i + 1
        except Exception as e:
            print(f"Error processing page {i}: {e}")
            import traceback
            traceback.print_exc()
    
    if cached_count > 0:
        print(f"📦 Cache: {cached_count}/{total} pages served from cache")
    emit_gemini_progress(total, total, f'✅ Hoàn tất! {total} trang ({cached_count} từ cache)')
    socketio.emit('translation_complete', {
        'session_id': session_id,
        'total': total
    })
    _active_jobs[session_id]['status'] = 'completed'

@app.route('/translate/status/<session_id>')
def translate_status(session_id):
    """Return current status of a translation job."""
    session_dir = os.path.join(RESULTS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    # Find completed result images (exclude _src_ and _orig_ temp files)
    completed = []
    for f in sorted(os.listdir(session_dir)):
        if f.endswith('.jpg') and not f.startswith('_'):
            name = f.replace('.jpg', '')
            completed.append({
                'name': name,
                'url': f'/static/results/{session_id}/{f}'
            })
    
    job_status = _active_jobs.get(session_id, {})
    
    return jsonify({
        'session_id': session_id,
        'status': job_status.get('status', 'unknown'),
        'total': job_status.get('total', 0),
        'completed': completed,
        'completed_count': len(completed),
        'error': job_status.get('error', '')
    })

@app.route('/translate/retry/<session_id>', methods=['POST'])
def translate_retry(session_id):
    """Retry a failed or hung translation job from where it left off."""
    session_dir = os.path.join(RESULTS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    # Find source files that haven't been translated yet
    completed_names = set()
    for f in os.listdir(session_dir):
        if f.endswith('.jpg') and not f.startswith('_'):
            completed_names.add(f.replace('.jpg', ''))
    
    # Find remaining source files
    remaining_files = []
    all_src_files = []
    for f in sorted(os.listdir(session_dir)):
        if f.startswith('_src_') and f.endswith('.png'):
            name = f[5:-4]  # Strip _src_ prefix and .png suffix
            all_src_files.append({
                'path': os.path.join(session_dir, f),
                'name': name,
                'original_path': os.path.join(session_dir, f'_orig_{name}.png')
            })
            if name not in completed_names:
                remaining_files.append(all_src_files[-1])
    
    if not remaining_files:
        return jsonify({'status': 'all_completed', 'total': len(all_src_files)})
    
    # Get job config from active_jobs or reconstruct from request
    source_lang = request.args.get('source_lang', 'zh')
    target_lang = request.args.get('target_lang', 'vi')
    upscale = request.args.get('upscale', 'false') == 'true'
    
    # Create a retry job with only remaining files
    retry_job = {
        'session_id': session_id,
        'session_dir': session_dir,
        'saved_files': remaining_files,
        'source_lang': source_lang,
        'target_lang': target_lang,
        'font_path': request.args.get('font_path', 'font/mangat.ttf'),
        'split_long_images': False,
        'upscale_image': upscale,
        'gemini_api_key': request.args.get('gemini_api_key', ''),
    }
    
    socketio.start_background_task(run_translation_job, retry_job)
    
    return jsonify({
        'status': 'retrying',
        'session_id': session_id,
        'remaining': len(remaining_files),
        'total': len(all_src_files)
    })

@app.route('/translate/retry-page/<session_id>/<page_name>', methods=['POST'])
def translate_retry_page(session_id, page_name):
    """Retry translation for a single page."""
    session_dir = os.path.join(RESULTS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    # Find the source file
    import urllib.parse
    page_name = urllib.parse.unquote(page_name)
    src_path = os.path.join(session_dir, f"_src_{page_name}.png")
    orig_path = os.path.join(session_dir, f"_orig_{page_name}.png")
    
    if not os.path.exists(src_path):
        return jsonify({'error': f'Source file not found: {page_name}'}), 404
    
    # Get settings from request
    source_lang = request.args.get('source_lang', 'zh')
    target_lang = request.args.get('target_lang', 'vi')
    upscale = request.args.get('upscale', 'false') == 'true'
    
    page_job = {
        'session_id': session_id,
        'session_dir': session_dir,
        'saved_files': [{
            'path': src_path,
            'name': page_name,
            'original_path': orig_path if os.path.exists(orig_path) else src_path
        }],
        'source_lang': source_lang,
        'target_lang': target_lang,
        'font_path': request.args.get('font_path', 'font/mangat.ttf'),
        'split_long_images': False,
        'upscale_image': upscale,
        'gemini_api_key': request.args.get('gemini_api_key', ''),
        '_single_page_retry': True,  # Flag for special emit
    }
    
    # Delete existing cache for this page so it re-processes
    import hashlib
    with open(src_path, 'rb') as f:
        img_hash = hashlib.md5(f.read()).hexdigest()
    settings = f"{source_lang}_{target_lang}_{upscale}_{page_job['font_path']}"
    settings_hash = hashlib.md5(settings.encode()).hexdigest()[:8]
    cache_path = os.path.join(RESULTS_DIR, '_cache', f"{img_hash}_{settings_hash}.jpg")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  🗑️ Cleared cache for {page_name}")
    
    socketio.start_background_task(run_translation_job, page_job)
    
    return jsonify({
        'status': 'retrying_page',
        'page_name': page_name,
        'session_id': session_id
    })

@app.route("/translate", methods=["POST"])
def upload_file():
    # Get translator selection
    translator_map = {
        "Opus-mt model": "hf",
        "NLLB": "nllb",
        "Gemini": "gemini",
        "Gemini Flash": "gemini",
        "Gemini Pro": "gemini-pro",
        "FreeLLM": "freellm",
        "Local LLM": "copilot"
    }
    selected_translator = translator_map.get(
        request.form["selected_translator"],
        request.form["selected_translator"].lower()
    )
    
    # Get Local LLM settings if selected (Ollama, LM Studio, etc.)
    copilot_server = request.form.get("copilot_server", "http://localhost:8080")
    copilot_model = request.form.get("copilot_model_input", "gpt-4o")
    
    # Get Gemini/FreeLLM API keys
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    freellm_api_key = request.form.get("freellm_api_key", "").strip()
    freellm_base_url = request.form.get("freellm_base_url", "http://127.0.0.1:31415/v1").strip()
    
    # Get context memory setting (checkbox - "on" if checked, None if not)
    use_context_memory = request.form.get("context_memory") == "on"

    # Get black bubble detection setting (checkbox - "on" if checked, None if not)
    enable_black_bubble = request.form.get("detect_black_bubbles") == "on"

    # Get split long images setting (checkbox - "on" if checked, None if not)
    split_long_images = request.form.get("split_long_images") == "on"

    # Get upscale setting (checkbox - "on" if checked, None if not)
    upscale_image = request.form.get("upscale_image") == "on"

    # Get font selection
    selected_font_raw = request.form["selected_font"]
    selected_font = selected_font_raw.lower()
    
    # Handle special font name mappings
    if selected_font == "auto (match original)":
        selected_font = "auto"
    elif selected_font == "animeace":
        selected_font = "animeace_"
    elif selected_font_raw.startswith("Yuki-"):
        # Keep original case for Yuki fonts
        selected_font = selected_font_raw

    # Get OCR engine
    selected_ocr = request.form.get("selected_ocr", "chrome-lens").lower()
    
    # Get source language
    source_lang_map = {
        "japanese (manga)": "ja",
        "chinese (manhua)": "zh",
        "korean (manhwa)": "ko",
        "english (comic)": "en"
    }
    selected_source = request.form.get("selected_source_lang", "Japanese (Manga)").lower()
    source_lang = source_lang_map.get(selected_source, "ja")
    
    # Get target language
    target_lang_map = {
        "english": "en",
        "vietnamese": "vi", 
        "chinese": "zh",
        "korean": "ko",
        "thai": "th",
        "indonesian": "id",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru"
    }
    selected_language = request.form.get("selected_language", "Vietnamese").lower()
    target_lang = target_lang_map.get(selected_language, "vi")
    
    # Get translation style/custom prompt
    style_map = {
        "default": "",
        "casual (thân mật)": "casual",
        "formal (trang trọng)": "formal",
        "keep honorifics (-san, senpai...)": "keep_honorifics",
        "web novel style": "web_novel",
        "action (ngắn gọn)": "action",
        "literal (sát nghĩa)": "literal",
        "custom...": ""
    }
    selected_style = request.form.get("selected_style", "Default").lower()
    style = style_map.get(selected_style, "")
    
    # Get custom prompt if provided
    custom_prompt = request.form.get("custom_prompt", "").strip()
    if custom_prompt:
        style = custom_prompt  # Override style with custom prompt

    # Get pipeline mode (Classic / Gemini Full / Gemini Hybrid)
    pipeline_mode_raw = request.form.get("selected_pipeline_mode", "Classic").lower()
    if "full" in pipeline_mode_raw or "banana" in pipeline_mode_raw:
        pipeline_mode = "gemini_full"
    elif "hybrid" in pipeline_mode_raw:
        pipeline_mode = "gemini_hybrid"
    else:
        pipeline_mode = "classic"
    
    # Get concurrent workers setting for Gemini pipeline
    gemini_workers = int(request.form.get("gemini_workers", "3"))
    gemini_workers = max(1, min(5, gemini_workers))  # Clamp 1-5
    
    print(f"Pipeline mode: {pipeline_mode} | Workers: {gemini_workers}")

    # Get multiple files
    files = request.files.getlist("files")
    
    if not files or files[0].filename == '':
        return redirect("/")
    
    # Initialize translator and OCR once for all images
    manga_translator = MangaTranslator(source=source_lang, target=target_lang)
    
    # Set custom prompt for Gemini
    if selected_translator in ("gemini", "gemini-pro") and style:
        manga_translator._gemini_custom_prompt = style
    
    # Set custom prompt for Local LLM
    if selected_translator == "copilot" and style:
        manga_translator._copilot_custom_prompt = style
    
    # Set Gemini API key
    # Set Gemini API key and model
    if selected_translator in ("gemini", "gemini-pro") and gemini_api_key:
        manga_translator._gemini_api_key = gemini_api_key
    if selected_translator == "gemini-pro":
        manga_translator._gemini_model = "gemini-2.5-pro"
    elif selected_translator == "gemini":
        manga_translator._gemini_model = "gemini-2.5-flash"

    if selected_translator == "freellm" and style:
        manga_translator._freellm_custom_prompt = style
    
    if selected_translator == "freellm" and freellm_api_key:
        manga_translator._freellm_api_key = freellm_api_key
        manga_translator._freellm_base_url = freellm_base_url
        print(f"Using FreeLLM API with provided key")
    
    # Set Copilot settings
    if selected_translator == "copilot":
        manga_translator._copilot_server = copilot_server
        manga_translator._copilot_model = copilot_model
        print(f"Using Local LLM: {copilot_server} / model: {copilot_model}")
    
    if selected_ocr == "paddleocr":
        if _OCR_CACHE.get("paddleocr") is None:
            from ocr.paddle_ocr import PaddleOcrEngine
            _OCR_CACHE["paddleocr"] = PaddleOcrEngine(ocr_language=source_lang)
        mocr = _OCR_CACHE["paddleocr"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "google-vision":
        if _OCR_CACHE.get("google_vision") is None:
            from ocr.google_vision_ocr import GoogleVisionOCR
            _OCR_CACHE["google_vision"] = GoogleVisionOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["google_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "freellm-vision":
        if _OCR_CACHE.get("freellm_vision") is None:
            from ocr.freellm_vision_ocr import FreeLLMVisionOCR
            _OCR_CACHE["freellm_vision"] = FreeLLMVisionOCR(
                api_key=freellm_api_key or os.environ.get("FREELLM_API_KEY"),
                base_url=freellm_base_url or os.environ.get("FREELLM_BASE_URL"),
                ocr_language=source_lang
            )
        mocr = _OCR_CACHE["freellm_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "gemini-vision":
        if _OCR_CACHE.get("gemini_vision") is None:
            from ocr.gemini_vision_ocr import GeminiVisionOCR
            _OCR_CACHE["gemini_vision"] = GeminiVisionOCR(
                api_key=gemini_api_key or os.environ.get("GEMINI_API_KEY"),
                ocr_language=source_lang
            )
        mocr = _OCR_CACHE["gemini_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "tesseract":
        if _OCR_CACHE.get("tesseract") is None:
            from ocr.tesseract_ocr import TesseractOCR
            _OCR_CACHE["tesseract"] = TesseractOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["tesseract"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "chrome-lens":
        if _OCR_CACHE["chrome_lens"] is None:
            _OCR_CACHE["chrome_lens"] = ChromeLensOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["chrome_lens"]
        if hasattr(mocr, 'ocr_language'):
            mocr.ocr_language = source_lang
    else:
        if _OCR_CACHE["manga_ocr"] is None:
            _OCR_CACHE["manga_ocr"] = MangaOcr()
        mocr = _OCR_CACHE["manga_ocr"]
    
    # Initialize font analyzer for auto font matching
    font_analyzer = None
    if selected_font == "auto":
        try:
            from font_analyzer import FontAnalyzer
            # Use same API key as Gemini translator
            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: No Gemini API key provided for font analysis")
            font_analyzer = FontAnalyzer(api_key=api_key)
            print("Font analyzer initialized for auto font matching")
        except Exception as e:
            print(f"Failed to initialize font analyzer: {e}")
            selected_font = "mangat"  # Fallback to default
    
    # Process all images
    processed_images = []
    auto_font_determined = False  # Flag to analyze font only once
    
    # ====================================================================
    # GEMINI PIPELINE MODE (Full / Hybrid) - New Banana Pipeline
    # ====================================================================
    if pipeline_mode in ["gemini_full", "gemini_hybrid"] and GEMINI_PIPELINE_AVAILABLE:
        strategy = "full" if pipeline_mode == "gemini_full" else "hybrid"
        print(f"\n{'='*50}")
        print(f"🍌 GEMINI BANANA PIPELINE ({strategy.upper()})")
        print(f"{'='*50}")
        
        # Initialize Gemini Pipeline
        try:
            pipeline = GeminiMangaPipeline(
                api_key=gemini_api_key or None,
                strategy=strategy,
                max_workers=gemini_workers,
                temperature=0.1
            )
        except Exception as e:
            print(f"⚠️ Gemini Pipeline init failed: {e}")
            print("Falling back to Classic pipeline...")
            pipeline_mode = "classic"  # Fallback
        
        if pipeline_mode != "classic":  # If not fallen back
            # Read all images
            all_images = []
            for file in files:
                if file and file.filename:
                    try:
                        file_stream = file.stream
                        file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if image is not None:
                            name = os.path.splitext(file.filename)[0]
                            all_images.append({'image': image, 'name': name})
                    except Exception as e:
                        print(f"Error reading {file.filename}: {e}")
            
            if not all_images:
                return redirect("/")
            
            total = len(all_images)
            
            def emit_gemini_progress(current, total_count, message):
                """Emit progress for Gemini pipeline."""
                try:
                    socketio.emit('progress', {
                        'phase': 'gemini',
                        'current': current,
                        'total': total_count,
                        'message': message,
                        'percent': int((current / max(total_count, 1)) * 100)
                    })
                except Exception:
                    pass
            
            emit_gemini_progress(0, total, f'🍌 Bắt đầu Gemini Pipeline ({strategy})...')
            
            if strategy == "full":
                # Strategy A: Gemini does EVERYTHING (OCR + Inpaint + Translate + Typeset)
                print(f"Processing {total} pages with Gemini Full (Banana 🍌)...")
                
                # Process batch with concurrent workers
                image_arrays = [img['image'] for img in all_images]
                results = pipeline.process_batch(
                    image_arrays, source_lang, target_lang,
                    progress_callback=emit_gemini_progress
                )
                
                # Save results to disk
                session_id = uuid.uuid4().hex[:12]
                session_dir = os.path.join(RESULTS_DIR, session_id)
                os.makedirs(session_dir, exist_ok=True)
                cleanup_old_results()
                
                for i, (result_img, img_data) in enumerate(zip(results, all_images)):
                    try:
                        if result_img is None:
                            result_img = img_data['image']  # Fallback to original
                        
                        base_name = img_data['name']
                        
                        # Split long images if enabled
                        if split_long_images:
                            chunks = split_long_image(result_img)
                        else:
                            chunks = [result_img]
                        
                        for j, chunk in enumerate(chunks):
                            chunk_name = f"{base_name}_part{j+1}" if len(chunks) > 1 else base_name
                            filename = f"{chunk_name}.jpg"
                            filepath = os.path.join(session_dir, filename)
                            cv2.imwrite(filepath, chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            processed_images.append({
                                "name": chunk_name,
                                "url": f"/static/results/{session_id}/{filename}"
                            })
                    except Exception as e:
                        print(f"Error saving result {i}: {e}")
                
                emit_gemini_progress(total, total, f'✅ Hoàn tất! {total} trang')
            
            else:
                # Gemini Hybrid: ASYNC mode
                # Save files to disk, return immediately, process in background
                session_id = uuid.uuid4().hex[:12]
                session_dir = os.path.join(RESULTS_DIR, session_id)
                os.makedirs(session_dir, exist_ok=True)
                cleanup_old_results()
                
                # Save uploaded images to temp dir for background processing
                saved_files = []
                for img_data in all_images:
                    temp_path = os.path.join(session_dir, f"_src_{img_data['name']}.png")
                    cv2.imwrite(temp_path, img_data['image'])
                    saved_files.append({
                        'path': temp_path,
                        'name': img_data['name'],
                        'original_path': os.path.join(session_dir, f"_orig_{img_data['name']}.png")
                    })
                    # Also save original (unprocessed) if different
                    if 'original' in img_data:
                        cv2.imwrite(saved_files[-1]['original_path'], img_data['original'])
                    else:
                        cv2.imwrite(saved_files[-1]['original_path'], img_data['image'])
                
                # Store job config for background processing
                _pending_jobs[session_id] = {
                    'session_id': session_id,
                    'session_dir': session_dir,
                    'saved_files': saved_files,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'font_path': get_font_path(selected_font),
                    'split_long_images': split_long_images,
                    'upscale_image': upscale_image,
                    'gemini_api_key': gemini_api_key,
                }
                
                return jsonify({'status': 'started', 'session_id': session_id, 'total': len(saved_files)})
            
            # Skip to rendering template if Gemini pipeline processed
            if processed_images:
                return render_template("translate.html", images=processed_images)
            else:
                print("⚠️ Gemini pipeline produced no results, falling back to Classic")
                pipeline_mode = "classic"
                # Re-read files (streams already consumed)
                # Note: This fallback won't work because file streams are consumed.
                # In production, we'd need to buffer the files first.
    
    # ====================================================================
    # CLASSIC PIPELINE (existing 4-phase processing)
    # ====================================================================
    # For Local LLM, Gemini and FreeLLM: Use multi-page batch processing
    if selected_translator in ["copilot", "gemini", "gemini-pro", "freellm"] and pipeline_mode == "classic":
        # First, read all images into memory
        all_images = []
        for file in files:
            if file and file.filename:
                try:
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    name = os.path.splitext(file.filename)[0]
                    all_images.append({'image': image, 'name': name})
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")
        
        if not all_images:
            return redirect("/")
        
        # Auto font: analyze first image
        if selected_font == "auto" and font_analyzer is not None:
            try:
                results = detect_bubbles(MODEL_PATH, all_images[0]['image'], enable_black_bubble)
                if results:
                    x1, y1, x2, y2 = results[0][:4]
                    first_bubble = all_images[0]['image'][int(y1):int(y2), int(x1):int(x2)]
                    selected_font = font_analyzer.analyze_and_match(first_bubble)
                    print(f"Auto font matched: {selected_font}")
                else:
                    selected_font = "mangat"
            except Exception as e:
                print(f"Font analysis failed: {e}")
                selected_font = "mangat"
        
        # Initialize translator based on type
        if selected_translator == "copilot":
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                # Get custom prompt for Local LLM
                copilot_custom_prompt = style if style else None
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model} (style: {style or 'default'})")
        
        elif selected_translator in ("gemini", "gemini-pro"):
            if not hasattr(manga_translator, '_gemini_translator') or manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = gemini_api_key or None  # Let GeminiTranslator handle fallback
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                gemini_model = getattr(manga_translator, '_gemini_model', 'gemini-2.5-flash')
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key,
                    custom_prompt=custom_prompt,
                    model=gemini_model
                )
                print(f"Gemini translator initialized with model: {gemini_model}")
        
        elif selected_translator == "freellm":
            if not hasattr(manga_translator, '_freellm_translator') or manga_translator._freellm_translator is None:
                from translator.freellm_translator import FreeLLMTranslator
                api_key = freellm_api_key
                base_url = freellm_base_url
                if not api_key:
                    api_key = os.environ.get("FREELLM_API_KEY")
                custom_prompt = getattr(manga_translator, '_freellm_custom_prompt', None)
                manga_translator._freellm_translator = FreeLLMTranslator(
                    api_key=api_key, 
                    base_url=base_url,
                    custom_prompt=custom_prompt
                )
                print("FreeLLM translator initialized for multi-page batching")
        
        # Process with multi-page batching (10 pages per API call)
        processed_results = process_images_with_batch(
            all_images, manga_translator, mocr, selected_font, 
            translator_type=selected_translator, batch_size=10,
            use_context_memory=use_context_memory,
            enable_black_bubble=enable_black_bubble,
            ocr_engine_name=selected_ocr,
            source_lang=source_lang,
            target_lang=target_lang,
            style=style
        )
        
        # Save results to disk (avoid massive base64 responses for large batches)
        session_id = uuid.uuid4().hex[:12]
        session_dir = os.path.join(RESULTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        cleanup_old_results()  # Clean up old sessions
        
        for result in processed_results:
            try:
                image = result['image']
                base_name = result['name']
                
                # Split long images if enabled
                if split_long_images:
                    chunks = split_long_image(image)
                else:
                    chunks = [image]
                
                # Save each chunk to disk
                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        chunk_name = f"{base_name}_part{i+1}"
                    else:
                        chunk_name = base_name
                    
                    filename = f"{chunk_name}.jpg"
                    filepath = os.path.join(session_dir, filename)
                    cv2.imwrite(filepath, chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    processed_images.append({
                        "name": chunk_name,
                        "url": f"/static/results/{session_id}/{filename}"
                    })
            except Exception as e:
                print(f"Error saving {result['name']}: {e}")
    
    else:
        # For other translators: Use per-image processing (original flow)
        for file in files:
            if file and file.filename:
                try:
                    # Read image
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    # Auto font: analyze FIRST image only
                    if selected_font == "auto" and font_analyzer is not None and not auto_font_determined:
                        try:
                            results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
                            if results:
                                x1, y1, x2, y2 = results[0][:4]
                                first_bubble = image[int(y1):int(y2), int(x1):int(x2)]
                                selected_font = font_analyzer.analyze_and_match(first_bubble)
                                print(f"Auto font matched (once for all images): {selected_font}")
                            else:
                                selected_font = "mangat"
                        except Exception as e:
                            print(f"Font analysis failed: {e}")
                            selected_font = "mangat"
                        auto_font_determined = True
                    
                    # Get original filename
                    name = os.path.splitext(file.filename)[0]
                    
                    # Process image
                    processed_image = process_single_image(
                        image, manga_translator, mocr, 
                        selected_translator, selected_font, None,
                        enable_black_bubble=enable_black_bubble
                    )
                    
                    # Split long images if enabled
                    if split_long_images:
                        chunks = split_long_image(processed_image)
                    else:
                        chunks = [processed_image]
                    
                    # Encode each chunk to base64
                    for i, chunk in enumerate(chunks):
                        _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
                        
                        # Add suffix if split into multiple chunks
                        if len(chunks) > 1:
                            chunk_name = f"{name}_part{i+1}"
                        else:
                            chunk_name = name
                        
                        processed_images.append({
                            "name": chunk_name,
                            "data": encoded_image
                        })
                    
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
                    continue
    
    if not processed_images:
        return redirect("/")
    
    return render_template("translate.html", images=processed_images)


def _sort_bubbles_manga_order(bubbles):
    """Sort bubbles in Japanese manga reading order: top-to-bottom rows,
    right-to-left within each row. Each bubble dict must have x1,y1,x2,y2.
    """
    if not bubbles:
        return []
    heights = [max(0.0, b['y2'] - b['y1']) for b in bubbles]
    avg_h = sum(heights) / len(heights) if heights else 0.0
    row_thresh = max(avg_h * 0.6, 1e-9)

    items = sorted(bubbles, key=lambda b: (b['y1'] + b['y2']) / 2)
    rows = []
    for b in items:
        yc = (b['y1'] + b['y2']) / 2
        if rows and abs(yc - rows[-1]['yc_mean']) <= row_thresh:
            row = rows[-1]
            row['items'].append(b)
            row['yc_mean'] = sum(((it['y1'] + it['y2']) / 2) for it in row['items']) / len(row['items'])
        else:
            rows.append({'yc_mean': yc, 'items': [b]})

    ordered = []
    for row in rows:
        row['items'].sort(key=lambda b: -((b['x1'] + b['x2']) / 2))
        ordered.extend(row['items'])
    return ordered


@app.route("/extract-text", methods=["POST"])
def extract_text():
    """OCR-only endpoint. Runs bubble detection + OCR on uploaded images and
    returns a single .txt file with texts grouped per page in Japanese manga
    reading order (right-to-left, top-to-bottom).
    """
    selected_ocr = request.form.get("selected_ocr", "chrome-lens").lower()
    enable_black_bubble = request.form.get("detect_black_bubbles") == "on"
    filter_sfx = request.form.get("filter_sfx", "on") == "on"
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    freellm_api_key = request.form.get("freellm_api_key", "").strip()
    freellm_base_url = request.form.get("freellm_base_url", "").strip()

    source_lang_map = {
        "japanese (manga)": "ja",
        "chinese (manhua)": "zh",
        "korean (manhwa)": "ko",
        "english (comic)": "en",
    }
    selected_source = request.form.get("selected_source_lang", "Japanese (Manga)").lower()
    source_lang = source_lang_map.get(selected_source, "ja")

    files = request.files.getlist("files")
    if not files or files[0].filename == '':
        return redirect("/")

    if selected_ocr == "paddleocr":
        if _OCR_CACHE.get("paddleocr") is None:
            from ocr.paddle_ocr import PaddleOcrEngine
            _OCR_CACHE["paddleocr"] = PaddleOcrEngine(ocr_language=source_lang)
        mocr = _OCR_CACHE["paddleocr"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "google-vision":
        if _OCR_CACHE.get("google_vision") is None:
            from ocr.google_vision_ocr import GoogleVisionOCR
            _OCR_CACHE["google_vision"] = GoogleVisionOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["google_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "freellm-vision":
        if _OCR_CACHE.get("freellm_vision") is None:
            from ocr.freellm_vision_ocr import FreeLLMVisionOCR
            _OCR_CACHE["freellm_vision"] = FreeLLMVisionOCR(
                api_key=freellm_api_key or os.environ.get("FREELLM_API_KEY"),
                base_url=freellm_base_url or os.environ.get("FREELLM_BASE_URL"),
                ocr_language=source_lang,
            )
        mocr = _OCR_CACHE["freellm_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "tesseract":
        if _OCR_CACHE.get("tesseract") is None:
            from ocr.tesseract_ocr import TesseractOCR
            _OCR_CACHE["tesseract"] = TesseractOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["tesseract"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "gemini-vision":
        if _OCR_CACHE.get("gemini_vision") is None:
            from ocr.gemini_vision_ocr import GeminiVisionOCR
            _OCR_CACHE["gemini_vision"] = GeminiVisionOCR(
                api_key=gemini_api_key or os.environ.get("GEMINI_API_KEY"),
                ocr_language=source_lang
            )
        mocr = _OCR_CACHE["gemini_vision"]
        mocr.ocr_language = source_lang
    elif selected_ocr == "chrome-lens":
        if _OCR_CACHE["chrome_lens"] is None:
            import asyncio as _asyncio
            try:
                _asyncio.get_event_loop()
            except RuntimeError:
                _asyncio.set_event_loop(_asyncio.new_event_loop())
            _OCR_CACHE["chrome_lens"] = ChromeLensOCR(ocr_language=source_lang)
        mocr = _OCR_CACHE["chrome_lens"]
        mocr.ocr_language = source_lang
    else:
        if _OCR_CACHE["manga_ocr"] is None:
            _OCR_CACHE["manga_ocr"] = MangaOcr()
        mocr = _OCR_CACHE["manga_ocr"]

    use_batch_ocr = hasattr(mocr, 'process_batch')
    use_full_page_ocr = hasattr(mocr, 'get_text_blocks') or hasattr(mocr, 'detect_and_recognize_blocks')

    def _flatten(t):
        return " ".join((t or "").split())

    def _char_in_lang(ch, lang):
        cp = ord(ch)
        if lang == "ja":
            # Hiragana + Katakana + CJK unified + half-width kana
            return (0x3040 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0xFF66 <= cp <= 0xFF9F)
        if lang == "zh":
            return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF)
        if lang == "ko":
            return (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F)
        if lang == "en":
            return ('A' <= ch <= 'Z') or ('a' <= ch <= 'z')
        return True

    def _keep_for_lang(text, lang):
        letters = [c for c in text if not c.isspace() and not c.isdigit() and not c in "、。，．・「」『』()（）!?！？…—-—:：;；\"'　"]
        if not letters:
            # Only punctuation/numbers/spaces — skip
            return False
        matched = sum(1 for c in letters if _char_in_lang(c, lang))
        # Keep if at least 40% of letter characters belong to target script
        return matched / len(letters) >= 0.4

    # Common English/Vietnamese-manga SFX and animal-sound words. Purely
    # heuristic — used only when the filter checkbox is on.
    _SFX_WORDS = {
        "ARF", "BARK", "WOOF", "GRR", "GRRR", "GROWL", "MEOW", "MOO", "OINK",
        "QUACK", "NEIGH", "BAA", "TWEET", "CHIRP", "HISS", "ROAR", "SQUEAK",
        "RUSTLE", "THUD", "THUMP", "BANG", "BOOM", "CRASH", "SLAM", "CLANG",
        "CLINK", "CLICK", "CLACK", "CRACK", "SNAP", "POP", "PUFF", "WHOOSH",
        "SWOOSH", "SWISH", "SPLASH", "SPLAT", "SLURP", "GULP", "GASP", "PANT",
        "SIGH", "GRUNT", "SNIFF", "SNORT", "SOB", "GIGGLE", "CHUCKLE", "HAHA",
        "HEHE", "AHAHA", "WHAM", "ZAP", "POW", "BAM", "TICK", "TOCK", "DING",
        "DONG", "BEEP", "HONK", "BUZZ", "SHH", "SHHH", "SHUSH", "PSST",
        "TAP", "PATTER", "STOMP", "CREAK", "CRUNCH", "WHIRR", "HUM", "RING",
        "RUMBLE", "ROAR", "GRIND", "SIZZLE", "FIZZ", "CRACKLE", "STOMP",
        "STEP", "DRIP", "SPLIT", "TCH", "TSK", "HMPH", "HMMPH", "HUFF",
        "HAAH", "AAH", "AAAH", "AAAAH", "OOF", "OW", "OWW", "OUCH", "ERR",
        "UM", "UMM", "UH", "UHH", "HUH", "GAK", "GHK", "GLK", "GAH", "AGH",
        "YEE", "YEEE", "WOW", "WOOW", "WHOA", "GASP", "PANT",
        # Japanese SFX romanised
        "GOSHI", "GORO", "KOTSU", "DOKI", "DOKUN", "BAKUN", "GATA", "GATAN",
        "MOGU", "MOGE", "GORI", "GARI", "GYU", "BATA", "GUNI", "GUSHA",
        "PIKA", "PACHI", "PYU", "SUU", "FUU", "HYUUU",
    }

    def _looks_like_sfx(text):
        if not text:
            return True
        # Strip surrounding punctuation, keep letters + digits + inner spaces
        cleaned = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text)
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            return True
        upper_tokens = [t.upper() for t in tokens]
        # Rule 1: single- or multi-token blocks where every token is repeated
        # the same SFX-looking word (ARF! ARF!, RUSTLE RUSTLE)
        if len(set(upper_tokens)) == 1 and upper_tokens[0] in _SFX_WORDS:
            return True
        # Rule 2: every token is in the SFX list
        if all(t in _SFX_WORDS for t in upper_tokens):
            return True
        # Rule 3: single short token, uppercase-only in source, no vowels
        if len(tokens) == 1:
            tok = tokens[0]
            if tok.isupper() and len(tok) <= 5 and not any(v in tok for v in "AEIOU"):
                return True
        # Rule 4: repeated-syllable garbage: "モ" "ゲ" "ク" — single-char CJK tokens
        if all(len(t) == 1 and not t.isascii() for t in tokens):
            return True
        return False

    def _gemini_filter_sfx(pages_in, api_key):
        """Ask Gemini to keep only meaningful dialogue lines. Returns pages
        with a filtered `texts` list. Falls back to input on any error.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            # Build a compact payload
            payload_lines = []
            for i, p in enumerate(pages_in, start=1):
                payload_lines.append(f"### page {i}")
                for j, t in enumerate(p['texts'], start=1):
                    payload_lines.append(f"{j}. {t}")
            payload = "\n".join(payload_lines)
            prompt = (
                "You are cleaning OCR output from a manga/comic. Below is a list of "
                "text blocks grouped per page. Remove entries that are onomatopoeia, "
                "sound effects (SFX), animal noises, or non-verbal grunts. Keep only "
                "meaningful dialogue/narration/thought. Preserve original text; do NOT "
                "translate or paraphrase. Return the result as JSON with this exact "
                "shape: {\"pages\":[{\"page\":1,\"texts\":[\"...\",\"...\"]}, ...]}. "
                "Include every page number even if its texts array is empty.\n\n"
                f"{payload}"
            )
            resp = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(resp.text)
            by_page = {int(x['page']): [t for t in x.get('texts', []) if t] for x in data.get('pages', [])}
            out = []
            for i, p in enumerate(pages_in, start=1):
                out.append({'name': p['name'], 'texts': by_page.get(i, p['texts'])})
            return out
        except Exception as e:
            print(f"Gemini SFX filter failed, using heuristic-only output: {e}")
            return pages_in

    pages = []
    for idx, file in enumerate(files, start=1):
        if not file or not file.filename:
            continue
        try:
            file_bytes = np.frombuffer(file.stream.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")
            continue

        name = os.path.splitext(file.filename)[0]
        socketio.emit('progress', {
            'phase': 'ocr', 'current': idx, 'total': len(files),
            'percent': int((idx - 1) / max(len(files), 1) * 100),
            'message': f'OCR page {idx}/{len(files)}: {name}'
        })

        page_texts = []
        if use_full_page_ocr:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                blocks = mocr.get_text_blocks(pil_img)
                items = []
                for blk in blocks:
                    txt = _flatten(blk.get('text', ''))
                    if not txt or not _keep_for_lang(txt, source_lang):
                        continue
                    g = blk.get('geometry', {}) or {}
                    cx = g.get('center_x', 0.0)
                    cy = g.get('center_y', 0.0)
                    w = g.get('width', 0.0)
                    h = g.get('height', 0.0)
                    items.append({
                        'text': txt,
                        'x1': cx - w / 2, 'x2': cx + w / 2,
                        'y1': cy - h / 2, 'y2': cy + h / 2,
                    })
                ordered = _sort_bubbles_manga_order(items)
                page_texts = [it['text'] for it in ordered]
            except Exception as e:
                print(f"Chrome Lens full-page OCR failed for {name}: {e}")
                page_texts = []
        else:
            results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
            bubbles = []
            for r in results:
                x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                crop = image[y1:y2, x1:x2]
                bubbles.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'img': Image.fromarray(crop),
                })
            ordered = _sort_bubbles_manga_order(bubbles)
            if ordered:
                imgs = [b['img'] for b in ordered]
                if use_batch_ocr:
                    try:
                        raw = mocr.process_batch(imgs)
                    except Exception as e:
                        print(f"Batch OCR failed, falling back: {e}")
                        raw = [mocr(im) for im in imgs]
                else:
                    raw = [mocr(im) for im in imgs]
                page_texts = [_flatten(t) for t in raw]

        page_texts = [t for t in page_texts if t and _keep_for_lang(t, source_lang)]
        if filter_sfx:
            page_texts = [t for t in page_texts if not _looks_like_sfx(t)]
        pages.append({'name': name, 'texts': page_texts})

    # Optional second-pass LLM filter when a Gemini key is provided
    if filter_sfx and gemini_api_key:
        socketio.emit('progress', {
            'phase': 'ocr', 'current': len(files), 'total': len(files),
            'percent': 99, 'message': 'Gemini filter: dropping SFX/noise...'
        })
        pages = _gemini_filter_sfx(pages, gemini_api_key)

    socketio.emit('progress', {
        'phase': 'done', 'current': len(files), 'total': len(files),
        'percent': 100, 'message': 'Text extraction complete'
    })

    lines = []
    for i, page in enumerate(pages, start=1):
        label = f"page {i}"
        if page['name']:
            label += f" ({page['name']})"
        lines.append(f"{label}:")
        lines.append("")
        if not page['texts']:
            lines.append("* (no text detected)")
        else:
            for t in page['texts']:
                lines.append(f"* {t}" if t else "* (empty)")
        lines.append("")

    buf = io.BytesIO(("\n".join(lines)).encode("utf-8"))
    return send_file(
        buf,
        mimetype='text/plain; charset=utf-8',
        as_attachment=True,
        download_name='manga_texts.txt',
    )


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Create and download a ZIP file containing all translated images."""
    try:
        images_data = request.form.get("images_data", "[]")
        images = json.loads(images_data)
        
        if not images:
            return redirect("/")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(images):
                name = img.get('name', f'image_{i+1}')
                data = img.get('data', '')
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(data)
                
                # Add to ZIP with proper filename
                filename = f"{name}_translated.png"
                zip_file.writestr(filename, image_bytes)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_translated.zip'
        )
    
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return redirect("/")


@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear translation cache and old results."""
    try:
        from translator.translation_cache import get_cache
        cache = get_cache()
        stats_before = cache.stats()
        cache.clear()
        cleanup_old_results(max_age_seconds=0)  # Remove all results
        return jsonify({
            "success": True,
            "message": f"Đã xóa {stats_before['count']} cache entries ({stats_before['size_mb']} MB)"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_frozen = getattr(sys, 'frozen', False)
    debug = not is_frozen and os.environ.get("FLASK_DEBUG", "0") == "1"

    if is_frozen:
        import threading
        import webbrowser
        def _open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=_open_browser, daemon=True).start()

    socketio.run(app, host="127.0.0.1", port=port, debug=debug)
