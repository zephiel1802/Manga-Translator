"""
Real-ESRGAN upscaler module for manga/anime images.

Supports two backends (in priority order):
  1. realesrgan-ncnn-vulkan CLI — portable binary, works on macOS/Linux/Windows
     without CUDA. Uses Vulkan (Metal on Mac) for GPU acceleration.
  2. Python realesrgan package — requires basicsr + CUDA. Used as fallback
     if already installed.

Supports 2x and 4x upscaling with automatic tiling for large images.
"""

import os
import sys
import platform
import subprocess
import tempfile
import zipfile
import stat
from typing import Optional, Tuple
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BIN_DIR = os.path.join(_SCRIPT_DIR, "bin")
_MODEL_DIR = os.path.join(_SCRIPT_DIR, "model")

# ── NCNN binary configuration ───────────────────────────────────────────────
_NCNN_VERSION = "20220424"
_NCNN_RELEASE_TAG = "v0.2.5.0"

# Binary name per platform
_NCNN_BIN_NAME = (
    "realesrgan-ncnn-vulkan.exe" if sys.platform == "win32"
    else "realesrgan-ncnn-vulkan"
)

# Download URL template
_NCNN_URLS = {
    "darwin":  f"https://github.com/xinntao/Real-ESRGAN/releases/download/{_NCNN_RELEASE_TAG}/realesrgan-ncnn-vulkan-{_NCNN_VERSION}-macos.zip",
    "linux":   f"https://github.com/xinntao/Real-ESRGAN/releases/download/{_NCNN_RELEASE_TAG}/realesrgan-ncnn-vulkan-{_NCNN_VERSION}-ubuntu.zip",
    "win32":   f"https://github.com/xinntao/Real-ESRGAN/releases/download/{_NCNN_RELEASE_TAG}/realesrgan-ncnn-vulkan-{_NCNN_VERSION}-windows.zip",
}

# Default model for anime/manga content
_NCNN_MODEL_NAME = "realesr-animevideov3"

# ── Python backend configuration ────────────────────────────────────────────
_PYTHON_MODEL_NAME = "RealESRGAN_x4plus_anime_6B"
_PYTHON_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
_PYTHON_MODEL_PATH = os.path.join(_MODEL_DIR, f"{_PYTHON_MODEL_NAME}.pth")

# Threshold for enabling tiling to avoid OOM errors
_LARGE_IMAGE_THRESHOLD = 2000
_TILE_SIZE = 400

# ── Detect available backends ────────────────────────────────────────────────

def _find_ncnn_binary() -> Optional[str]:
    """Return absolute path to realesrgan-ncnn-vulkan binary, or None."""
    # Check project bin/ directory first
    local_path = os.path.join(_BIN_DIR, _NCNN_BIN_NAME)
    if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
        return local_path
    # Check system PATH
    import shutil
    system_path = shutil.which(_NCNN_BIN_NAME)
    return system_path


def _check_python_backend() -> bool:
    """Check if the Python realesrgan package is importable."""
    try:
        import importlib
        try:
            importlib.import_module('torchvision.transforms.functional_tensor')
        except ModuleNotFoundError:
            import types
            import torchvision.transforms.functional as F
            fake_module = types.ModuleType('torchvision.transforms.functional_tensor')
            fake_module.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules['torchvision.transforms.functional_tensor'] = fake_module

        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401
        return True
    except Exception:
        return False


# Cache detection results at module load
_NCNN_BIN_PATH = _find_ncnn_binary()
_PYTHON_AVAILABLE = _check_python_backend()

# ── Auto-download NCNN binary ───────────────────────────────────────────────

def _download_ncnn_binary() -> Optional[str]:
    """
    Download and extract the realesrgan-ncnn-vulkan binary for the current
    platform into the bin/ directory. Returns the binary path on success,
    or None on failure.
    """
    plat = sys.platform
    url = _NCNN_URLS.get(plat)
    if url is None:
        print(f"[Upscaler] No NCNN binary available for platform: {plat}")
        return None

    os.makedirs(_BIN_DIR, exist_ok=True)
    zip_path = os.path.join(_BIN_DIR, "realesrgan-ncnn-vulkan.zip")

    print(f"[Upscaler] Downloading realesrgan-ncnn-vulkan for {plat}...")
    print(f"[Upscaler] URL: {url}")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"[Upscaler] Download failed: {e}")
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        return None

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_BIN_DIR)
        os.remove(zip_path)
    except Exception as e:
        print(f"[Upscaler] Extraction failed: {e}")
        return None

    # The zip usually contains a subdirectory; find the binary
    bin_path = None
    for root, _dirs, files in os.walk(_BIN_DIR):
        if _NCNN_BIN_NAME in files:
            bin_path = os.path.join(root, _NCNN_BIN_NAME)
            break

    if bin_path is None:
        print(f"[Upscaler] Binary not found after extraction.")
        return None

    # Make executable
    st = os.stat(bin_path)
    os.chmod(bin_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"[Upscaler] Binary ready: {bin_path}")
    return bin_path


def ensure_ncnn_binary() -> Optional[str]:
    """
    Ensure the NCNN binary is available. Downloads if necessary.
    Returns the binary path or None.
    """
    global _NCNN_BIN_PATH
    if _NCNN_BIN_PATH is not None:
        return _NCNN_BIN_PATH
    _NCNN_BIN_PATH = _download_ncnn_binary()
    return _NCNN_BIN_PATH


# ── NCNN CLI Backend ─────────────────────────────────────────────────────────

class _NCNNUpscaler:
    """Upscaler backend using realesrgan-ncnn-vulkan CLI."""

    def __init__(self, bin_path: str, scale: int = 2):
        self._bin_path = bin_path
        self._scale = scale
        # Find model directory relative to binary (shipped with the zip)
        bin_dir = os.path.dirname(bin_path)
        models_dir = os.path.join(bin_dir, "models")
        self._models_dir = models_dir if os.path.isdir(models_dir) else None

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale a BGR numpy image via CLI subprocess."""
        import cv2

        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_in:
            input_path = f_in.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
            output_path = f_out.name

        try:
            cv2.imwrite(input_path, img)

            cmd = [
                self._bin_path,
                "-i", input_path,
                "-o", output_path,
                "-n", _NCNN_MODEL_NAME,
                "-s", str(self._scale),
                "-f", "png",
            ]
            if self._models_dir:
                cmd.extend(["-m", self._models_dir])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                print(f"[Upscaler] NCNN CLI error (code {result.returncode}): {stderr}")
                return img

            # Read output
            output = cv2.imread(output_path, cv2.IMREAD_COLOR)
            if output is None:
                print("[Upscaler] Failed to read NCNN output image.")
                return img

            return output

        except subprocess.TimeoutExpired:
            print("[Upscaler] NCNN CLI timed out after 300s.")
            return img
        except Exception as e:
            print(f"[Upscaler] NCNN upscale failed: {e}")
            return img
        finally:
            # Cleanup temp files
            for p in (input_path, output_path):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ── Python Backend ───────────────────────────────────────────────────────────

class _PythonUpscaler:
    """Upscaler backend using the Python realesrgan package."""

    def __init__(self, scale: int = 2):
        self._scale = scale
        self._upsampler = None

    def _ensure_model_weights(self) -> str:
        """Download model weights if not already present."""
        if os.path.isfile(_PYTHON_MODEL_PATH):
            return _PYTHON_MODEL_PATH

        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"[Upscaler] Downloading {_PYTHON_MODEL_NAME} weights...")

        try:
            import urllib.request
            urllib.request.urlretrieve(_PYTHON_MODEL_URL, _PYTHON_MODEL_PATH)
            print("[Upscaler] Download complete.")
        except Exception as e:
            if os.path.isfile(_PYTHON_MODEL_PATH):
                os.remove(_PYTHON_MODEL_PATH)
            raise RuntimeError(f"Failed to download model weights: {e}") from e

        return _PYTHON_MODEL_PATH

    def _detect_device(self) -> tuple:
        """Auto-detect CUDA / CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                print("[Upscaler] Using CUDA device.")
                return 0, True
        except ImportError:
            pass
        print("[Upscaler] Using CPU device.")
        return None, False

    def _init_model(self, tile: int = 0):
        """Initialize the Real-ESRGAN upsampler model."""
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = self._ensure_model_weights()
        gpu_id, use_half = self._detect_device()

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4,
        )

        self._upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model,
            tile=tile, tile_pad=10, pre_pad=0,
            half=use_half, gpu_id=gpu_id,
        )
        print(f"[Upscaler] Python model initialized (tile={tile}).")

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale a BGR numpy image using the Python backend."""
        h, w = img.shape[:2]
        tile = _TILE_SIZE if (h > _LARGE_IMAGE_THRESHOLD or w > _LARGE_IMAGE_THRESHOLD) else 0

        if self._upsampler is None:
            self._init_model(tile=tile)
        elif self._upsampler.tile != tile:
            self._init_model(tile=tile)

        try:
            output, _ = self._upsampler.enhance(img, outscale=self._scale)
            return output
        except Exception as e:
            print(f"[Upscaler] Python enhancement failed: {e}")
            if tile == 0:
                print(f"[Upscaler] Retrying with tiling (tile={_TILE_SIZE})...")
                self._init_model(tile=_TILE_SIZE)
                try:
                    output, _ = self._upsampler.enhance(img, outscale=self._scale)
                    return output
                except Exception as retry_e:
                    print(f"[Upscaler] Retry also failed: {retry_e}.")
                    return img
            return img


# ── Public API (unchanged interface) ─────────────────────────────────────────

class MangaUpscaler:
    """
    Wraps Real-ESRGAN for manga/anime upscaling.

    Uses realesrgan-ncnn-vulkan CLI (preferred, works on macOS/Linux without
    CUDA) or falls back to the Python realesrgan package.

    The model is lazy-loaded on first use and reused for subsequent calls.

    Args:
        scale: Output scale factor (2 or 4). Default is 2 for speed.
    """

    def __init__(self, scale: int = 2):
        if scale not in (2, 4):
            raise ValueError(f"Unsupported scale factor: {scale}. Must be 2 or 4.")
        self._scale = scale
        self._backend = None  # Lazy-loaded
        self._backend_name = None

    @staticmethod
    def is_available() -> bool:
        """Check if any upscaling backend is available."""
        # Try to ensure NCNN binary (download if needed)
        if ensure_ncnn_binary() is not None:
            return True
        return _PYTHON_AVAILABLE

    @staticmethod
    def backend_name() -> str:
        """Return the name of the active backend."""
        if _NCNN_BIN_PATH is not None:
            return "ncnn-vulkan"
        if _PYTHON_AVAILABLE:
            return "python-realesrgan"
        return "none"

    def _init_backend(self):
        """Initialize the best available backend."""
        bin_path = ensure_ncnn_binary()
        if bin_path is not None:
            self._backend = _NCNNUpscaler(bin_path, scale=self._scale)
            self._backend_name = "ncnn-vulkan"
            print(f"[Upscaler] Using NCNN-Vulkan backend (scale={self._scale}x)")
            return

        if _PYTHON_AVAILABLE:
            self._backend = _PythonUpscaler(scale=self._scale)
            self._backend_name = "python-realesrgan"
            print(f"[Upscaler] Using Python Real-ESRGAN backend (scale={self._scale}x)")
            return

        raise RuntimeError(
            "No upscaler backend available. Install realesrgan-ncnn-vulkan "
            "or the Python realesrgan package."
        )

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale a manga/anime image using Real-ESRGAN.

        Args:
            img: Input image as OpenCV BGR numpy array (H, W, 3) uint8.

        Returns:
            Upscaled image as OpenCV BGR numpy array (H*scale, W*scale, 3) uint8.
            If no backend is available, returns the original image unchanged.
        """
        if self._backend is None:
            try:
                self._init_backend()
            except RuntimeError as e:
                print(f"[Upscaler] WARNING: {e}")
                return img

        return self._backend.upscale(img)


# ── Singleton ────────────────────────────────────────────────────────────────

_instance = None


def get_manga_upscaler(scale: int = 2) -> MangaUpscaler:
    """Get singleton instance of MangaUpscaler."""
    global _instance
    if _instance is None or _instance._scale != scale:
        _instance = MangaUpscaler(scale=scale)
    return _instance
