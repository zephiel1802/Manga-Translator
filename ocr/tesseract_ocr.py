"""
Tesseract OCR module for manga/manhua text recognition.
Supports vertical CJK text via --psm 5 mode.
Requires: brew install tesseract tesseract-lang (macOS) or apt install tesseract-ocr (Linux)
         pip install pytesseract
"""
import os
import numpy as np
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class TesseractOCR:
    """
    OCR engine using Tesseract with vertical CJK text support.
    
    Uses --psm 5 (vertical aligned text) for CJK languages,
    which correctly reads right-to-left columns top-to-bottom.
    """
    
    # Map BCP 47 language codes to Tesseract language codes
    LANG_MAP = {
        "zh": "chi_tra",       # Traditional Chinese
        "ja": "jpn_vert",      # Japanese vertical (preferred for manga)
        "ko": "kor",           # Korean
        "en": "eng",           # English
    }
    
    # Fallback if _vert variant not available
    LANG_FALLBACK = {
        "jpn_vert": "jpn",
    }
    
    # PSM modes for different text layouts
    # 5 = Assume a single uniform block of vertically aligned text
    # 6 = Assume a single uniform block of text
    # 4 = Assume a single column of text of variable sizes
    CJK_VERTICAL_PSM = 5
    HORIZONTAL_PSM = 6
    
    def __init__(self, ocr_language: str = "zh"):
        """
        Initialize Tesseract OCR.
        
        Args:
            ocr_language: BCP 47 language code (default: "zh")
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract is not installed. Install with: pip install pytesseract\n"
                "Also install Tesseract engine:\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "  Linux: apt install tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-jpn"
            )
        
        # Verify Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract engine not found. Install with:\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "  Linux: apt install tesseract-ocr"
            )
        
        self.ocr_language = ocr_language
        self._verify_language()
        print(f"[Tesseract OCR] Initialized (lang={ocr_language} → {self._get_tess_lang()})")
    
    def _get_tess_lang(self) -> str:
        """Get Tesseract language code from BCP 47 code."""
        tess_lang = self.LANG_MAP.get(self.ocr_language, "eng")
        
        # Check if the language is available, try fallback
        try:
            available = pytesseract.get_languages()
            if tess_lang not in available:
                fallback = self.LANG_FALLBACK.get(tess_lang)
                if fallback and fallback in available:
                    return fallback
                # Try without _vert suffix
                base_lang = tess_lang.replace("_vert", "")
                if base_lang in available:
                    return base_lang
        except Exception:
            pass
        
        return tess_lang
    
    def _verify_language(self):
        """Verify the language pack is installed."""
        try:
            available = pytesseract.get_languages()
            tess_lang = self._get_tess_lang()
            if tess_lang not in available:
                print(f"[Tesseract OCR] WARNING: Language '{tess_lang}' not found!")
                print(f"  Available: {', '.join(available)}")
                print(f"  Install with: brew install tesseract-lang (macOS)")
        except Exception as e:
            print(f"[Tesseract OCR] Could not verify languages: {e}")
    
    def _get_config(self) -> str:
        """Get Tesseract config for current language."""
        is_vertical_cjk = self.ocr_language in ("zh", "ja", "ko")
        psm = self.CJK_VERTICAL_PSM if is_vertical_cjk else self.HORIZONTAL_PSM
        return f"--psm {psm} --oem 3"
    
    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Increase contrast using simple thresholding
        # This helps with manga text which is usually black on white
        import numpy as np
        arr = np.array(image)
        
        # Adaptive threshold: if mostly white, use lower threshold
        mean_val = arr.mean()
        threshold = min(180, max(100, int(mean_val * 0.7)))
        
        # Apply binary threshold
        arr = ((arr < threshold) * 0).astype(np.uint8) + ((arr >= threshold) * 255).astype(np.uint8)
        
        return Image.fromarray(arr)
    
    def __call__(self, image) -> str:
        """
        OCR a single image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            str: Extracted text
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tess_lang = self._get_tess_lang()
        config = self._get_config()
        
        try:
            # Try with preprocessing first
            processed = self._preprocess(image)
            text = pytesseract.image_to_string(
                processed, 
                lang=tess_lang, 
                config=config
            ).strip()
            
            # If preprocessing gave empty result, try original
            if not text:
                text = pytesseract.image_to_string(
                    image,
                    lang=tess_lang,
                    config=config
                ).strip()
            
            # Clean up: remove extra whitespace/newlines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return text
        except Exception as e:
            print(f"[Tesseract OCR] Error: {e}")
            return ""
    
    def process_batch(self, images: list) -> list:
        """
        Process multiple images (sequential, Tesseract is local so fast).
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extracted texts
        """
        results = []
        for img in images:
            text = self(img)
            results.append(text)
        return results
