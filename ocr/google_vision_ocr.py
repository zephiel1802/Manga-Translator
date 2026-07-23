"""
Google Cloud Vision OCR module for manga/manhua text recognition.
Uses DOCUMENT_TEXT_DETECTION for excellent vertical CJK text support.
Best-in-class accuracy for Traditional Chinese, Japanese, Korean.

Requires: pip install google-cloud-vision
          A service account JSON credentials file
"""
import os
import io
import numpy as np
from PIL import Image

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False


class GoogleVisionOCR:
    """
    OCR engine using Google Cloud Vision API with DOCUMENT_TEXT_DETECTION.
    
    Excellent at reading vertical CJK text in manga/manhua/manhwa.
    Uses block-level detection to preserve reading order of text columns.
    """
    
    # Default path for credentials file
    DEFAULT_CREDENTIALS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "google_vision_credentials.json"
    )
    
    def __init__(self, ocr_language: str = "zh", credentials_path: str = None):
        """
        Initialize Google Cloud Vision OCR.
        
        Args:
            ocr_language: BCP 47 language code (default: "zh")
            credentials_path: Path to service account JSON file.
                              If None, uses google_vision_credentials.json in project root,
                              or GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        if not GOOGLE_VISION_AVAILABLE:
            raise ImportError(
                "google-cloud-vision is not installed. Install with:\n"
                "  pip install google-cloud-vision"
            )
        
        self.ocr_language = ocr_language
        
        # Set up credentials
        self._setup_credentials(credentials_path)
        
        # Initialize client
        self._client = vision.ImageAnnotatorClient()
        print(f"[Google Vision OCR] Initialized (lang={ocr_language})")
    
    def _setup_credentials(self, credentials_path: str = None):
        """Set up Google Cloud credentials."""
        # Priority: explicit path > default file > env var
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"[Google Vision OCR] Using credentials: {credentials_path}")
        elif os.path.exists(self.DEFAULT_CREDENTIALS_PATH):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.DEFAULT_CREDENTIALS_PATH
            print(f"[Google Vision OCR] Using credentials: {self.DEFAULT_CREDENTIALS_PATH}")
        elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            print(f"[Google Vision OCR] Using env credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        else:
            raise FileNotFoundError(
                "Google Cloud Vision credentials not found!\n"
                "Please either:\n"
                "  1. Place 'google_vision_credentials.json' in the project root\n"
                "  2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "  3. Pass credentials_path to constructor"
            )
    
    def _image_to_bytes(self, image) -> bytes:
        """Convert PIL Image or numpy array to PNG bytes."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed (remove alpha channel)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()
    
    # Language hints for Google Vision API
    LANG_HINTS = {
        "zh": ["zh-Hant"],       # Traditional Chinese
        "ja": ["ja"],            # Japanese
        "ko": ["ko"],            # Korean
        "en": ["en"],            # English
    }
    
    def _extract_text_from_response(self, response) -> str:
        """
        Extract text from Vision API response using symbol-level positions.
        
        For vertical CJK text, we don't trust Google's paragraph/word ordering
        because it often gets confused with small bubble crops. Instead, we:
        1. Extract all symbols with their bounding box positions
        2. Group symbols into columns (similar X center)
        3. Sort columns right-to-left, symbols top-to-bottom within each column
        4. Concatenate in correct reading order
        
        Args:
            response: Vision API annotate response
            
        Returns:
            str: Extracted text with proper reading order
        """
        if response.error.message:
            raise Exception(f'Google Vision API error: {response.error.message}')
        
        annotation = response.full_text_annotation
        if not annotation or not annotation.text:
            return ""
        
        is_vertical = self.ocr_language in ("zh", "ja", "ko")
        
        # Collect all symbols with their positions from all blocks
        all_symbols = []
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            vertices = symbol.bounding_box.vertices
                            cx = sum(v.x for v in vertices) / 4
                            cy = sum(v.y for v in vertices) / 4
                            w = abs(vertices[1].x - vertices[0].x) if vertices[1].x != vertices[0].x else abs(vertices[2].x - vertices[3].x)
                            h = abs(vertices[3].y - vertices[0].y) if vertices[3].y != vertices[0].y else abs(vertices[2].y - vertices[1].y)
                            
                            all_symbols.append({
                                "text": symbol.text,
                                "cx": cx,
                                "cy": cy,
                                "w": max(w, 1),
                                "h": max(h, 1),
                                "vertices": [(v.x, v.y) for v in vertices],
                            })
        
        if not all_symbols:
            return ""
        
        if is_vertical:
            return self._sort_vertical_cjk(all_symbols)
        else:
            return self._sort_horizontal(all_symbols)
    
    def _sort_vertical_cjk(self, symbols: list) -> str:
        """
        Sort symbols for vertical CJK reading: right-to-left columns, top-to-bottom.
        
        Algorithm:
        1. Calculate average character width as column grouping threshold
        2. Sort all symbols by X descending (rightmost first)
        3. Group symbols into columns (symbols with similar X centers)
        4. Within each column, sort top-to-bottom by Y
        5. Concatenate all characters
        
        Args:
            symbols: List of symbol dicts with cx, cy, text, w, h
            
        Returns:
            str: Text in correct vertical reading order
        """
        # Calculate average character width for column grouping
        avg_w = sum(s["w"] for s in symbols) / len(symbols)
        # Column threshold: symbols within this X distance are in the same column
        col_threshold = max(avg_w * 0.7, 8)
        
        # Sort by X descending (right to left)
        symbols.sort(key=lambda s: -s["cx"])
        
        # Group into columns
        columns = []
        current_col = [symbols[0]]
        
        for s in symbols[1:]:
            # Compare with the average X of the current column
            col_avg_cx = sum(c["cx"] for c in current_col) / len(current_col)
            if abs(s["cx"] - col_avg_cx) < col_threshold:
                current_col.append(s)
            else:
                columns.append(current_col)
                current_col = [s]
        columns.append(current_col)
        
        # Within each column, sort top-to-bottom by Y
        # Then concatenate characters
        result_parts = []
        for col in columns:
            col.sort(key=lambda s: s["cy"])
            col_text = ''.join(s["text"] for s in col)
            result_parts.append(col_text)
        
        return ''.join(result_parts)
    
    def _sort_horizontal(self, symbols: list) -> str:
        """
        Sort symbols for horizontal reading: top-to-bottom rows, left-to-right.
        
        Args:
            symbols: List of symbol dicts with cx, cy, text, w, h
            
        Returns:
            str: Text in correct horizontal reading order
        """
        avg_h = sum(s["h"] for s in symbols) / len(symbols)
        row_threshold = max(avg_h * 0.7, 8)
        
        # Sort by Y ascending (top to bottom)
        symbols.sort(key=lambda s: s["cy"])
        
        # Group into rows
        rows = []
        current_row = [symbols[0]]
        
        for s in symbols[1:]:
            row_avg_cy = sum(c["cy"] for c in current_row) / len(current_row)
            if abs(s["cy"] - row_avg_cy) < row_threshold:
                current_row.append(s)
            else:
                rows.append(current_row)
                current_row = [s]
        rows.append(current_row)
        
        # Within each row, sort left-to-right by X
        result_parts = []
        for row in rows:
            row.sort(key=lambda s: s["cx"])
            row_text = ''.join(s["text"] for s in row)
            result_parts.append(row_text)
        
        return ''.join(result_parts)
    
    def __call__(self, image) -> str:
        """
        OCR a single image (bubble crop).
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            str: Extracted text in correct reading order
        """
        try:
            content = self._image_to_bytes(image)
            vision_image = vision.Image(content=content)
            
            # Add language hints to improve text direction detection
            hints = self.LANG_HINTS.get(self.ocr_language, [])
            image_context = vision.ImageContext(language_hints=hints)
            
            # Use DOCUMENT_TEXT_DETECTION for structured text (better for comics)
            response = self._client.document_text_detection(
                image=vision_image,
                image_context=image_context,
            )
            
            return self._extract_text_from_response(response)
            
        except Exception as e:
            print(f"[Google Vision OCR] Error: {e}")
            return ""
    
    def process_batch(self, images: list) -> list:
        """
        Process multiple bubble images.
        Uses sequential calls (Google Vision API has its own rate limiting).
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extracted texts
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images):
            if i > 0 and i % 10 == 0:
                print(f"    Google Vision OCR progress: {i}/{total}")
            text = self(img)
            results.append(text)
        
        return results
