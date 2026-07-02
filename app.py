from flask import Flask, render_template, request, redirect, send_file, jsonify
from flask_socketio import SocketIO, emit
import io
import zipfile
import json
import warnings
import os
import sys

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detect_bubbles import detect_bubbles
from process_bubble import process_bubble, process_bubble_auto, is_dark_bubble, get_bubble_background_color, get_dominant_color, process_bubble_preserve_gradient
from translator.translator import MangaTranslator
from translator.context_memory import ContextMemory
from add_text import add_text
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
    results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
    
    if not results:
        return image
    
    # Phase 1: Collect all bubble data and OCR texts
    bubble_data = []
    texts_to_translate = []
    first_bubble_image = None  # For font analysis
    
    for result in results:
        # Handle both old format (6 items) and new format (7 items with is_dark_bubble)
        if len(result) >= 7:
            x1, y1, x2, y2, score, class_id, is_dark = result[:7]
        else:
            x1, y1, x2, y2, score, class_id = result[:6]
            is_dark = 0
        
        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Save first bubble for font analysis (before processing)
        if first_bubble_image is None:
            first_bubble_image = detected_image.copy()
        
        # Fix: detected_image is already uint8, no need to multiply by 255
        im = Image.fromarray(detected_image)
        text = mocr(im)
        
        # Use auto detection or forced dark based on detection flag
        detected_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
        
        bubble_data.append({
            'detected_image': detected_image,
            'contour': cont,
            'coords': (int(x1), int(y1), int(x2), int(y2)),
            'is_dark': bubble_is_dark,
            'fill_color': detected_color
        })
        texts_to_translate.append(text)
    
    # Phase 2: Batch translate
    if selected_translator == "gemini" and len(texts_to_translate) > 1:
        # Use batch translation for Gemini
        try:
            if manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = getattr(manga_translator, '_gemini_api_key', None)
                if not api_key:
                    raise ValueError("Gemini API key not provided")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key, 
                    custom_prompt=custom_prompt
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
        # Use white text for dark bubbles, black text for light bubbles
        text_color = (255, 255, 255) if data.get('is_dark', False) else (0, 0, 0)
        add_text(data['detected_image'], translated_text, font_path, data['contour'], text_color)
    
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


def process_images_with_batch(images_data, manga_translator, mocr, selected_font, translator_type, batch_size=10, use_context_memory=True, enable_black_bubble=True):
    """
    Process multiple images with multi-page batching for Copilot or Gemini.
    Collects all texts first, batch translates, then applies translations.
    
    Args:
        images_data: List of dicts with 'image', 'name' keys
        manga_translator: MangaTranslator instance with translator
        mocr: OCR engine
        selected_font: Font to use
        translator_type: 'copilot' or 'gemini'
        batch_size: Number of pages per API call
        use_context_memory: Whether to include context from all pages for better translation
        
    Returns:
        List of processed images with translations applied
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
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
        
        results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
        if not results:
            all_pages_data[name] = {'image': image, 'bubbles': [], 'texts': []}
            print(f" - 0 bubbles")
            continue
        
        print(f" - {len(results)} bubbles")
        
        bubble_data = []
        
        for bubble_idx, result in enumerate(results):
            # Handle both old format (6 items) and new format (7 items with is_dark_bubble)
            if len(result) >= 7:
                x1, y1, x2, y2, score, class_id, is_dark = result[:7]
            else:
                x1, y1, x2, y2, score, class_id = result[:6]
                is_dark = 0
            
            detected_image = image[int(y1):int(y2), int(x1):int(x2)]
            
            # IMPORTANT: Add to OCR queue BEFORE processing (which fills white/black)
            all_bubble_images.append(Image.fromarray(detected_image.copy()))
            bubble_mapping.append((name, bubble_idx))
            
            # Process bubble (fill with auto-detected or specified color based on type)
            processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
            
            bubble_data.append({
                'detected_image': processed_image,
                'contour': cont,
                'coords': (int(x1), int(y1), int(x2), int(y2)),
                'is_dark': bubble_is_dark,
                'fill_color': detected_color
            })
        
        all_pages_data[name] = {
            'image': image,
            'bubbles': bubble_data,
            'texts': []  # Will fill after OCR
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
        
        # Map texts back to pages
        for (page_name, bubble_idx), text in zip(bubble_mapping, all_texts):
            all_pages_data[page_name]['texts'].append(text)
        
        ocr_time = time.time() - ocr_start
        print(f"({ocr_time:.1f}s)")
        print(f"✓ OCR completed in {ocr_time:.1f}s ({len(all_bubble_images)/ocr_time:.1f} bubbles/sec)")
        emit_progress('ocr', 1, 1, f'OCR hoàn tất ({len(all_bubble_images)} bubbles)')
    
    # Phase 3: Batch translate all pages together
    emit_progress('translation', 0, 1, 'Đang dịch...')
    pages_texts = {name: data['texts'] for name, data in all_pages_data.items() if data['texts']}
    all_translations = {}
    
    if pages_texts:
        # Get the translator based on type
        if translator_type == "copilot" and hasattr(manga_translator, '_local_llm_translator') and manga_translator._local_llm_translator:
            translator = manga_translator._local_llm_translator
            translator_name = "Local LLM"
        elif translator_type == "gemini" and hasattr(manga_translator, '_gemini_translator') and manga_translator._gemini_translator:
            translator = manga_translator._gemini_translator
            translator_name = "Gemini"
        else:
            translator = None
            translator_name = "Unknown"
        
        if translator:
            print(f"{translator_name} batch translating {len(pages_texts)} pages in chunks of {batch_size}...")
            
            # Initialize context memory if enabled
            context_memory = None
            if use_context_memory:
                context_memory = ContextMemory()
                print(f"  Context Memory enabled - tracking terms and story context")
            
            # Process in batches
            page_names = list(pages_texts.keys())
            
            for i in range(0, len(page_names), batch_size):
                batch_names = page_names[i:i + batch_size]
                batch_texts = {name: pages_texts[name] for name in batch_names}
                
                print(f"  Translating batch {i//batch_size + 1}: pages {i+1}-{min(i+batch_size, len(page_names))}")
                
                try:
                    translated = translator.translate_pages_batch(
                        batch_texts,
                        source=manga_translator.source,
                        target=manga_translator.target,
                        context_memory=context_memory
                    )
                    all_translations.update(translated)
                    
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
                        except:
                            all_translations[name] = texts  # Return original on error
    
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
        
        # Apply text to bubbles on the ORIGINAL image
        for bubble, text in zip(bubbles, translated_texts):
            x1, y1, x2, y2 = bubble['coords']
            # Get the region in the original image (this is a view, modifications affect original)
            bubble_region = image[y1:y2, x1:x2]
            # Use white text for dark bubbles, black text for light bubbles
            text_color = (255, 255, 255) if bubble.get('is_dark', False) else (0, 0, 0)
            # Add translated text
            add_text(bubble_region, text, font_path, bubble['contour'], text_color)
        
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


@app.route("/translate", methods=["POST"])
def upload_file():
    # Get translator selection
    translator_map = {
        "Opus-mt model": "hf",
        "NLLB": "nllb",
        "Gemini": "gemini",
        "Local LLM": "copilot"  # copilot is internal name for OpenAI-compatible endpoints
    }
    selected_translator = translator_map.get(
        request.form["selected_translator"],
        request.form["selected_translator"].lower()
    )
    
    # Get Local LLM settings if selected (Ollama, LM Studio, etc.)
    copilot_server = request.form.get("copilot_server", "http://localhost:8080")
    copilot_model = request.form.get("copilot_model_input", "gpt-4o")
    
    # Get Gemini API key from form
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    
    # Get context memory setting (checkbox - "on" if checked, None if not)
    use_context_memory = request.form.get("context_memory") == "on"

    # Get black bubble detection setting (checkbox - "on" if checked, None if not)
    enable_black_bubble = request.form.get("detect_black_bubbles") == "on"

    # Get split long images setting (checkbox - "on" if checked, None if not)
    split_long_images = request.form.get("split_long_images") == "on"

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

    # Get multiple files
    files = request.files.getlist("files")
    
    if not files or files[0].filename == '':
        return redirect("/")
    
    # Initialize translator and OCR once for all images
    manga_translator = MangaTranslator(source=source_lang, target=target_lang)
    
    # Set custom prompt for Gemini
    if selected_translator == "gemini" and style:
        manga_translator._gemini_custom_prompt = style
    
    # Set custom prompt for Local LLM
    if selected_translator == "copilot" and style:
        manga_translator._copilot_custom_prompt = style
    
    # Set Gemini API key
    if selected_translator == "gemini" and gemini_api_key:
        manga_translator._gemini_api_key = gemini_api_key
        print(f"Using Gemini API with provided key")
    
    # Set Copilot settings
    if selected_translator == "copilot":
        manga_translator._copilot_server = copilot_server
        manga_translator._copilot_model = copilot_model
        print(f"Using Local LLM: {copilot_server} / model: {copilot_model}")
    
    if selected_ocr == "chrome-lens":
        if _OCR_CACHE["chrome_lens"] is None:
            _OCR_CACHE["chrome_lens"] = ChromeLensOCR()
        mocr = _OCR_CACHE["chrome_lens"]
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
            selected_font = "animeace_"  # Fallback to default
    
    # Process all images
    processed_images = []
    auto_font_determined = False  # Flag to analyze font only once
    
    # For Local LLM and Gemini: Use multi-page batch processing
    if selected_translator in ["copilot", "gemini"]:
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
                    x1, y1, x2, y2, _, _ = results[0]
                    first_bubble = all_images[0]['image'][int(y1):int(y2), int(x1):int(x2)]
                    selected_font = font_analyzer.analyze_and_match(first_bubble)
                    print(f"Auto font matched: {selected_font}")
                else:
                    selected_font = "animeace_"
            except Exception as e:
                print(f"Font analysis failed: {e}")
                selected_font = "animeace_"
        
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
        
        elif selected_translator == "gemini":
            if not hasattr(manga_translator, '_gemini_translator') or manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key required. Please enter it in the web form.")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key,
                    custom_prompt=custom_prompt
                )
                print("Gemini translator initialized for multi-page batching")
        
        # Process with multi-page batching (10 pages per API call)
        processed_results = process_images_with_batch(
            all_images, manga_translator, mocr, selected_font, 
            translator_type=selected_translator, batch_size=10,
            use_context_memory=use_context_memory,
            enable_black_bubble=enable_black_bubble
        )
        
        # Encode results to base64 (with optional splitting)
        for result in processed_results:
            try:
                image = result['image']
                base_name = result['name']
                
                # Split long images if enabled
                if split_long_images:
                    chunks = split_long_image(image)
                else:
                    chunks = [image]
                
                # Encode each chunk
                for i, chunk in enumerate(chunks):
                    _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
                    
                    # Add suffix if split into multiple chunks
                    if len(chunks) > 1:
                        chunk_name = f"{base_name}_part{i+1}"
                    else:
                        chunk_name = base_name
                    
                    processed_images.append({
                        "name": chunk_name,
                        "data": encoded_image
                    })
            except Exception as e:
                print(f"Error encoding {result['name']}: {e}")
    
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
                                x1, y1, x2, y2, _, _ = results[0]
                                first_bubble = image[int(y1):int(y2), int(x1):int(x2)]
                                selected_font = font_analyzer.analyze_and_match(first_bubble)
                                print(f"Auto font matched (once for all images): {selected_font}")
                            else:
                                selected_font = "animeace_"
                        except Exception as e:
                            print(f"Font analysis failed: {e}")
                            selected_font = "animeace_"
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

    if selected_ocr == "chrome-lens":
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
    use_full_page_ocr = selected_ocr == "chrome-lens"

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
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
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


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
