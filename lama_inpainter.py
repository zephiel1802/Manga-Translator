"""
Lama inpainting module for the Manga Translator.
Provides a wrapper around SimpleLama for neural network based image inpainting.
"""

import cv2
import numpy as np
from PIL import Image
import logging

try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False


class LamaInpainter:
    """Wrapper class for the SimpleLama inpainting model."""

    def __init__(self):
        """Initializes the LamaInpainter. The model is lazy-loaded."""
        self._model = None

    def _ensure_model(self):
        """Ensures that the SimpleLama model is loaded."""
        if self._model is None:
            if not LAMA_AVAILABLE:
                raise ImportError("simple_lama_inpainting is not installed.")
            print("Loading SimpleLama model...")
            self._model = SimpleLama()
            print("SimpleLama model loaded successfully.")

    def inpaint(self, image_cv: np.ndarray, mask_cv: np.ndarray) -> np.ndarray:
        """
        Inpaints an image using a mask.

        Args:
            image_cv: The source image as an OpenCV BGR numpy array.
            mask_cv: The binary mask as a numpy array where 255 indicates areas to inpaint.

        Returns:
            The inpainted image as an OpenCV BGR numpy array.
        """
        self._ensure_model()

        # Convert OpenCV BGR to PIL RGB
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert mask to PIL L (grayscale)
        if len(mask_cv.shape) > 2:
            mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
        pil_mask = Image.fromarray(mask_cv).convert('L')

        # Run inpainting
        result_pil = self._model(pil_image, pil_mask)

        # Ensure output size matches input
        if result_pil.size != pil_image.size:
            result_pil = result_pil.resize(pil_image.size, Image.Resampling.LANCZOS)

        # Convert back to OpenCV BGR
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def inpaint_region(self, full_image_cv: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                       mask_cv: np.ndarray, fade_radius: int = 5) -> np.ndarray:
        """
        Inpaints a specific region of an image with padding for context and blends the result back.

        Args:
            full_image_cv: The full source image.
            x1, y1, x2, y2: Bounding box of the region to inpaint.
            mask_cv: Mask corresponding to the bounding box region.
            fade_radius: Radius for Gaussian blur on the mask edges for blending.

        Returns:
            The modified full image.
        """
        self._ensure_model()

        img_h, img_w = full_image_cv.shape[:2]

        # Add padding
        padding = 20
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(img_w, x2 + padding)
        py2 = min(img_h, y2 + padding)

        # Extract padded region from image
        region_img = full_image_cv[py1:py2, px1:px2].copy()

        # Create padded mask
        region_h, region_w = py2 - py1, px2 - px1
        padded_mask = np.zeros((region_h, region_w), dtype=np.uint8)
        
        # Calculate where the original mask goes inside the padded mask
        mx1 = x1 - px1
        my1 = y1 - py1
        
        # Ensure sizes match before assignment
        mask_h, mask_w = mask_cv.shape[:2]
        padded_mask[my1:my1+mask_h, mx1:mx1+mask_w] = mask_cv

        # Run inpainting on the region
        inpainted_region = self.inpaint(region_img, padded_mask)

        # Blend back into full image using alpha composite
        result_image = full_image_cv.copy()

        # Create alpha mask for blending (soft edges)
        alpha_mask = padded_mask.copy().astype(np.float32) / 255.0
        
        if fade_radius > 0:
            # Blur the alpha mask to create a soft transition
            blur_size = fade_radius * 2 + 1
            alpha_mask = cv2.GaussianBlur(alpha_mask, (blur_size, blur_size), 0)
        
        # Expand alpha mask to 3 channels for broadcasting
        alpha_mask = np.expand_dims(alpha_mask, axis=2)

        # Composite the inpainted region over the original region
        original_region = result_image[py1:py2, px1:px2]
        blended_region = (inpainted_region * alpha_mask + original_region * (1.0 - alpha_mask)).astype(np.uint8)
        
        # Put back into full image
        result_image[py1:py2, px1:px2] = blended_region

        return result_image

    def is_available(self) -> bool:
        """Returns True if the simple_lama_inpainting library is available."""
        return LAMA_AVAILABLE


_instance = None


def get_lama_inpainter() -> LamaInpainter:
    """Returns a singleton instance of the LamaInpainter."""
    global _instance
    if _instance is None:
        _instance = LamaInpainter()
    return _instance
