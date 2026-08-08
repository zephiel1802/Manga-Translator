"""
PaddleOCR module for manga/manhua text recognition.
Best-in-class for Chinese Traditional vertical text.
Developed by Baidu, specialized for CJK languages.

Requires: pip install paddlepaddle paddleocr
Supports PaddleOCR v3.x (paddleocr >= 3.0)
"""
import os
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class PaddleOcrEngine:
    """
    OCR engine using PaddleOCR with excellent vertical CJK support.
    
    Auto-detects text direction (vertical/horizontal) and reads
    Traditional Chinese, Japanese, Korean with high accuracy,
    even for handwritten or comic-style fonts.
    
    Compatible with PaddleOCR v3.x API.
    """
    
    # Map BCP 47 codes to PaddleOCR language codes
    LANG_MAP = {
        "zh": "chinese_cht",   # Traditional Chinese (best for manhua)
        "ja": "japan",         # Japanese
        "ko": "korean",       # Korean
        "en": "en",           # English
    }
    
    def __init__(self, ocr_language: str = "zh"):
        """
        Initialize PaddleOCR engine.
        
        Args:
            ocr_language: BCP 47 language code (default: "zh")
        """
        if not PADDLE_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Install with:\n"
                "  pip install paddlepaddle paddleocr\n"
                "  (GPU: pip install paddlepaddle-gpu paddleocr)"
            )
        
        self.ocr_language = ocr_language
        self._ocr_instance = None
        self._current_lang = None
        self._init_ocr()
        
    def _init_ocr(self):
        """Initialize or re-initialize PaddleOCR with current language."""
        paddle_lang = self.LANG_MAP.get(self.ocr_language, "chinese_cht")
        
        if self._ocr_instance is None or self._current_lang != paddle_lang:
            print(f"[PaddleOCR] Loading model (lang={paddle_lang})...")
            # PaddleOCR v3.x
            self._ocr_instance = PaddleOCR(
                lang=paddle_lang,
            )
            self._current_lang = paddle_lang
            print(f"[PaddleOCR] Ready!")
    
    def _sort_results_for_reading(self, polys, texts, scores):
        """
        Sort OCR results in correct reading order.
        For vertical CJK: right-to-left columns, top-to-bottom within column.
        For horizontal text: top-to-bottom rows, left-to-right within row.
        
        Args:
            polys: list of polygon boxes (each is list of 4 [x,y] points)
            texts: list of recognized text strings
            scores: list of confidence scores
            
        Returns:
            tuple: (sorted_polys, sorted_texts, sorted_scores)
        """
        if not polys:
            return [], [], []
        
        # Create indexed tuples for sorting
        items = list(zip(polys, texts, scores))
        
        is_vertical = self.ocr_language in ("zh", "ja", "ko")
        
        if is_vertical:
            # For vertical text: sort by x descending (right→left), then y ascending (top→bottom)
            # Group by columns: boxes with similar center x belong to the same column
            def get_cx_cy(box):
                cx = (box[0][0] + box[2][0]) / 2
                cy = (box[0][1] + box[2][1]) / 2
                return cx, cy

            # Calculate average box width to use as column grouping threshold
            avg_width = sum((box[1][0] - box[0][0] + box[2][0] - box[3][0])/2 for box, _, _ in items) / len(items)
            threshold = max(avg_width * 0.6, 10) # At least 10 pixels

            # Sort primarily by X descending
            items.sort(key=lambda item: -get_cx_cy(item[0])[0])
            
            # Group into columns
            columns = []
            current_col = []
            last_cx = None
            
            for item in items:
                cx, cy = get_cx_cy(item[0])
                if last_cx is None or abs(cx - last_cx) < threshold:
                    current_col.append(item)
                else:
                    columns.append(current_col)
                    current_col = [item]
                last_cx = cx
            if current_col:
                columns.append(current_col)
                
            # Sort each column by Y ascending
            sorted_items = []
            for col in columns:
                col.sort(key=lambda item: get_cx_cy(item[0])[1])
                sorted_items.extend(col)
            items = sorted_items
        else:
            # For horizontal text: sort by y ascending, then x ascending
            def get_cx_cy(box):
                cx = (box[0][0] + box[2][0]) / 2
                cy = (box[0][1] + box[2][1]) / 2
                return cx, cy

            avg_height = sum((box[3][1] - box[0][1] + box[2][1] - box[1][1])/2 for box, _, _ in items) / len(items)
            threshold = max(avg_height * 0.6, 10)

            items.sort(key=lambda item: get_cx_cy(item[0])[1])
            
            rows = []
            current_row = []
            last_cy = None
            
            for item in items:
                cx, cy = get_cx_cy(item[0])
                if last_cy is None or abs(cy - last_cy) < threshold:
                    current_row.append(item)
                else:
                    rows.append(current_row)
                    current_row = [item]
                last_cy = cy
            if current_row:
                rows.append(current_row)
                
            sorted_items = []
            for row in rows:
                row.sort(key=lambda item: get_cx_cy(item[0])[0])
                sorted_items.extend(row)
            items = sorted_items
        
        sorted_polys = [item[0] for item in items]
        sorted_texts = [item[1] for item in items]
        sorted_scores = [item[2] for item in items]
        return sorted_polys, sorted_texts, sorted_scores
    
    def __call__(self, image) -> str:
        """
        OCR a single image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            str: Extracted text in correct reading order
        """
        # Ensure language model is current
        self._init_ocr()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            # PaddleOCR v3.x: use predict() instead of deprecated ocr()
            # Returns list of OCRResult objects (one per input image)
            results = self._ocr_instance.predict(
                image,
            )
            
            if not results:
                return ""
            
            # v3.x: each result is an OCRResult (dict-like) with:
            #   rec_texts: list of recognized text strings
            #   rec_scores: list of confidence scores
            #   rec_polys: list of polygon bounding boxes
            result = results[0]
            
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            rec_polys = result.get("rec_polys", [])
            
            if not rec_texts:
                return ""
            
            # Sort by reading order
            rec_polys, rec_texts, rec_scores = self._sort_results_for_reading(
                rec_polys, rec_texts, rec_scores
            )
            
            # Extract text, join with newlines
            texts = []
            for text in rec_texts:
                text = text.strip() if isinstance(text, str) else str(text).strip()
                if text:
                    texts.append(text)
            
            return '\n'.join(texts)
            
        except Exception as e:
            print(f"[PaddleOCR] Error: {e}")
            return ""
    
    def process_batch(self, images: list) -> list:
        """
        Process multiple images sequentially.
        PaddleOCR is local and fast, no need for concurrent processing.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extracted texts
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images):
            if i > 0 and i % 10 == 0:
                print(f"    PaddleOCR progress: {i}/{total}")
            text = self(img)
            results.append(text)
        
        return results
