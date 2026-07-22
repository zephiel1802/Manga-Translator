"""
Bridge module to use PanelCleanerZ's Comic Text Detector in Manga-Translator.

Provides text detection (bounding boxes), pixel-level text masks,
and text cleaning (mask fitting + LaMa inpainting) from PanelCleanerZ.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

# Add PanelCleanerZ to path
PCLEANER_ROOT = Path(__file__).parent.parent / "PanelCleanerZ"
if not PCLEANER_ROOT.exists():
    # Try common locations
    for candidate in [
        Path(r"c:\Users\zephi\Downloads\VibeCodes\PanelCleanerZ"),
        Path(__file__).parent / "PanelCleanerZ",
    ]:
        if candidate.exists():
            PCLEANER_ROOT = candidate
            break

if str(PCLEANER_ROOT) not in sys.path:
    sys.path.insert(0, str(PCLEANER_ROOT))


# Default model path
DEFAULT_CTD_MODEL = Path(__file__).parent / "model" / "comictextdetector.pt"


class PanelCleanerBridge:
    """
    Bridge between PanelCleanerZ's Comic Text Detector and Manga-Translator.
    
    Provides:
    - Text block detection with bounding boxes
    - Pixel-level text mask generation (U-Net)  
    - Smart text cleaning using mask fitting + LaMa inpainting
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the bridge with the Comic Text Detector model."""
        self._model = None
        self._lama = None
        self.model_path = model_path or str(DEFAULT_CTD_MODEL)
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _ensure_model(self):
        """Lazy-load the TextDetector model."""
        if self._model is not None:
            return
            
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Comic Text Detector model not found at {self.model_path}. "
                "Download from: https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"
            )
        
        print(f"Loading Comic Text Detector model from {self.model_path}...")
        from pcleaner.comic_text_detector.inference import TextDetector
        self._model = TextDetector(
            model_path=self.model_path,
            input_size=1024,
            device=self.device
        )
        print(f"Comic Text Detector loaded on {self.device}!")
    
    def _ensure_lama(self):
        """Lazy-load LaMa inpainting model."""
        if self._lama is not None:
            return True
        try:
            from lama_inpainter import get_lama_inpainter
            inpainter = get_lama_inpainter()
            if inpainter.is_available():
                self._lama = inpainter
                return True
        except Exception as e:
            print(f"LaMa not available: {e}")
        return False
        
    def detect_text_blocks(self, image_cv: np.ndarray):
        """
        Detect text blocks and generate pixel-level text mask.
        
        Args:
            image_cv: OpenCV BGR image
            
        Returns:
            tuple: (text_blocks, text_mask, mask_refined)
            - text_blocks: list of dicts with keys:
                - 'coords': (x1, y1, x2, y2) bounding box
                - 'language': 'ja', 'eng', or 'unknown'
                - 'vertical': bool
                - 'font_size': float
                - 'fg_color': (r, g, b) foreground/text color
                - 'bg_color': (r, g, b) background color
            - text_mask: raw binary text mask (uint8, 0 or 255)
            - mask_refined: refined text mask (uint8, 0 or 255)
        """
        self._ensure_model()
        
        from pcleaner.comic_text_detector.utils.textmask import REFINEMASK_ANNOTATION
        
        mask, mask_refined, blk_list = self._model(
            image_cv, 
            refine_mode=REFINEMASK_ANNOTATION, 
            keep_undetected_mask=True
        )
        
        text_blocks = []
        for blk in blk_list:
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            
            # Get colors (accumulated values, need to average by number of lines)
            num_lines = max(len(blk.lines), 1)
            fg_color = (
                int(blk.fg_r / num_lines) if blk.fg_r else 0,
                int(blk.fg_g / num_lines) if blk.fg_g else 0, 
                int(blk.fg_b / num_lines) if blk.fg_b else 0
            )
            bg_color = (
                int(blk.bg_r / num_lines) if blk.bg_r else 255,
                int(blk.bg_g / num_lines) if blk.bg_g else 255,
                int(blk.bg_b / num_lines) if blk.bg_b else 255
            )
            
            text_blocks.append({
                'coords': (x1, y1, x2, y2),
                'language': blk.language,
                'vertical': blk.vertical,
                'font_size': blk.font_size,
                'fg_color': fg_color,
                'bg_color': bg_color,
            })
        
        return text_blocks, mask, mask_refined
    
    def clean_image(self, image_cv: np.ndarray, mask_refined: np.ndarray, text_blocks: list):
        """
        Clean text from image using the refined mask.
        
        Uses two strategies:
        1. For uniform backgrounds (bubbles): fill with median border color
        2. For complex backgrounds: LaMa neural inpainting
        
        Args:
            image_cv: Original image (BGR)
            mask_refined: Refined text mask from detect_text_blocks
            text_blocks: Text blocks from detect_text_blocks
            
        Returns:
            cleaned_image: Image with all text removed
        """
        cleaned = image_cv.copy()
        h, w = cleaned.shape[:2]
        
        # Ensure mask is proper binary
        _, mask_binary = cv2.threshold(mask_refined, 127, 255, cv2.THRESH_BINARY)
        
        # Strategy 1: Try mask fitting for each text block
        # For blocks where the background is uniform, fill with median color
        lama_mask = np.zeros((h, w), dtype=np.uint8)  # Collect regions for LaMa
        
        for block in text_blocks:
            x1, y1, x2, y2 = block['coords']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Get the mask region for this block
            block_mask = mask_binary[y1:y2, x1:x2]
            block_img = cleaned[y1:y2, x1:x2]
            
            if np.sum(block_mask) == 0:
                continue
            
            # Dilate the mask slightly to cover edge artifacts
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            block_mask_dilated = cv2.dilate(block_mask, dilate_kernel, iterations=1)
            
            # Check if background is uniform by analyzing border pixels
            bg_color, is_uniform = self._analyze_block_background(block_img, block_mask_dilated)
            
            if is_uniform:
                # Simple fill with background color
                block_img[block_mask_dilated > 0] = bg_color
                cleaned[y1:y2, x1:x2] = block_img
            else:
                # Mark for LaMa inpainting
                lama_mask[y1:y2, x1:x2] = block_mask_dilated
        
        # Strategy 2: LaMa inpainting for complex backgrounds
        if np.sum(lama_mask) > 0:
            if self._ensure_lama():
                # Dilate LaMa mask more for better results
                lama_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                lama_mask = cv2.dilate(lama_mask, lama_dilate, iterations=2)
                cleaned = self._lama.inpaint(cleaned, lama_mask)
            else:
                # Fallback: Gaussian blur for regions that need inpainting
                blur_kernel = (15, 15)
                blurred = cv2.GaussianBlur(cleaned, blur_kernel, 0)
                # Only apply blur where lama_mask is active
                lama_mask_3ch = np.stack([lama_mask]*3, axis=-1) / 255.0
                cleaned = (blurred * lama_mask_3ch + cleaned * (1 - lama_mask_3ch)).astype(np.uint8)
        
        return cleaned
    
    def _analyze_block_background(self, block_img, block_mask):
        """
        Analyze if a text block has a uniform background.
        
        Returns:
            (bg_color, is_uniform): BGR color tuple and whether background is uniform
        """
        h, w = block_img.shape[:2]
        
        # Get border pixels (pixels just outside the mask)
        larger_mask = cv2.dilate(block_mask, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 
                                iterations=2)
        border_ring = cv2.subtract(larger_mask, block_mask)
        
        # Get colors at border pixels
        border_pixels = block_img[border_ring > 0]
        
        if len(border_pixels) < 10:
            # Not enough border pixels, sample from non-mask area
            non_mask = block_img[block_mask == 0]
            if len(non_mask) < 10:
                return (255, 255, 255), True
            border_pixels = non_mask
        
        # Calculate standard deviation of border colors
        std_dev = np.std(border_pixels.astype(np.float64), axis=0).mean()
        
        # If std_dev is low, background is uniform
        is_uniform = std_dev < 25.0
        
        # Calculate median color
        median_color = np.median(border_pixels, axis=0).astype(np.uint8)
        
        # Round near-white to pure white
        if all(c > 230 for c in median_color):
            median_color = np.array([255, 255, 255], dtype=np.uint8)
        
        return tuple(int(c) for c in median_color), is_uniform
    
    def detect_and_clean(self, image_cv: np.ndarray):
        """
        Full pipeline: detect text blocks + clean image.
        
        Args:
            image_cv: OpenCV BGR image
            
        Returns:
            dict with:
                - 'cleaned_image': Image with text removed
                - 'text_blocks': List of detected text block dicts
                - 'mask': Raw text mask
                - 'mask_refined': Refined text mask
        """
        text_blocks, mask, mask_refined = self.detect_text_blocks(image_cv)
        cleaned_image = self.clean_image(image_cv, mask_refined, text_blocks)
        
        return {
            'cleaned_image': cleaned_image,
            'text_blocks': text_blocks,
            'mask': mask,
            'mask_refined': mask_refined,
        }


# Singleton
_instance = None

def get_pcleaner_bridge() -> PanelCleanerBridge:
    """Get singleton instance of PanelCleanerBridge."""
    global _instance
    if _instance is None:
        _instance = PanelCleanerBridge()
    return _instance
