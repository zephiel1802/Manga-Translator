#!/bin/bash
echo "==================================================="
echo "Manga Translator - Startup Script (macOS/Linux)"
echo "==================================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# =============================================================
# System Dependencies (Tesseract OCR)
# =============================================================
if ! command -v tesseract &>/dev/null; then
    echo "[INFO] Tesseract OCR not found. Installing..."
    if command -v brew &>/dev/null; then
        brew install tesseract tesseract-lang
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-jpn
    else
        echo "[WARNING] Cannot auto-install Tesseract. Please install manually:"
        echo "  macOS:  brew install tesseract tesseract-lang"
        echo "  Linux:  apt install tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-jpn"
    fi
else
    # Check if Chinese Traditional language pack is installed
    if ! tesseract --list-langs 2>&1 | grep -q "chi_tra"; then
        echo "[INFO] Tesseract Chinese Traditional language pack not found. Installing..."
        if command -v brew &>/dev/null; then
            brew install tesseract-lang
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y tesseract-ocr-chi-tra tesseract-ocr-jpn
        fi
    fi
fi

# =============================================================
# Dependency Check - auto-detect and install missing packages
# =============================================================
echo "[INFO] Checking dependencies..."

# List of critical packages to verify (import_name:pip_name)
# Includes both Manga-Translator's own deps and PanelCleanerZ deps
# used via pcleaner_bridge.py
PACKAGES=(
    "flask:Flask"
    "flask_socketio:flask-socketio"
    "cv2:opencv-python"
    "PIL:pillow"
    "torch:torch"
    "torchvision:torchvision"
    "numpy:numpy"
    "tqdm:tqdm"
    "deep_translator:deep-translator"
    "translators:translators"
    "manga_ocr:manga-ocr"
    "ultralytics:ultralytics"
    "safetensors:safetensors"
    "cryptography:cryptography"
    "sentencepiece:sentencepiece"
    "werkzeug:Werkzeug"
    "engineio:python-engineio"
    "socketio:python-socketio"
    "huggingface_hub:huggingface-hub"
    "google.genai:google-genai"
    "openai:openai"
    "google.protobuf:protobuf"
    "pytesseract:pytesseract"
    "paddleocr:paddleocr"
    "gunicorn:gunicorn"
    "simple_lama_inpainting:simple-lama-inpainting"
    # PanelCleanerZ dependencies (used by pcleaner_bridge.py)
    "pyclipper:pyclipper"
    "shapely:shapely"
    "scipy:scipy"
    "loguru:loguru"
    "packaging:packaging"
)

MISSING=()

for entry in "${PACKAGES[@]}"; do
    IFS=':' read -r import_name pip_name <<< "$entry"
    if ! python3 -c "import $import_name" 2>/dev/null; then
        MISSING+=("$pip_name")
        echo "  [!] Missing: $pip_name ($import_name)"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "[INFO] Installing ${#MISSING[@]} missing package(s): ${MISSING[*]}"
    pip install "${MISSING[@]}"
    echo ""

    # Verify installation
    STILL_MISSING=()
    for entry in "${PACKAGES[@]}"; do
        IFS=':' read -r import_name pip_name <<< "$entry"
        if ! python3 -c "import $import_name" 2>/dev/null; then
            STILL_MISSING+=("$pip_name")
        fi
    done

    if [ ${#STILL_MISSING[@]} -gt 0 ]; then
        echo "[WARNING] Some packages could not be installed: ${STILL_MISSING[*]}"
        echo "[WARNING] Try installing manually: pip install ${STILL_MISSING[*]}"
        echo ""
        read -p "Continue anyway? (y/N): " choice
        if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
            echo "Aborted."
            exit 1
        fi
    else
        echo "[OK] All missing packages installed successfully!"
    fi
else
    echo "[OK] All dependencies are satisfied."
fi

# Also run pip install for any new additions to requirements.txt
# (uses --quiet to reduce noise, only installs what's missing)
echo "[INFO] Syncing with requirements.txt..."
pip install -q -r requirements.txt 2>/dev/null

# =============================================================
# Real-ESRGAN NCNN Binary (for image upscaling)
# =============================================================
NCNN_BIN="$SCRIPT_DIR/bin/realesrgan-ncnn-vulkan"
if [ ! -x "$NCNN_BIN" ]; then
    echo ""
    echo "[INFO] realesrgan-ncnn-vulkan not found. Downloading..."

    NCNN_VERSION="20220424"
    NCNN_TAG="v0.2.5.0"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        NCNN_PLATFORM="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        NCNN_PLATFORM="ubuntu"
    else
        echo "[WARNING] Unsupported platform for NCNN binary: $OSTYPE"
        echo "[WARNING] Upscale feature will not be available."
        NCNN_PLATFORM=""
    fi

    if [ -n "$NCNN_PLATFORM" ]; then
        NCNN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/${NCNN_TAG}/realesrgan-ncnn-vulkan-${NCNN_VERSION}-${NCNN_PLATFORM}.zip"
        NCNN_ZIP="$SCRIPT_DIR/bin/realesrgan-ncnn-vulkan.zip"

        mkdir -p "$SCRIPT_DIR/bin"
        echo "[INFO] Downloading from: $NCNN_URL"

        if curl -fSL -o "$NCNN_ZIP" "$NCNN_URL" 2>/dev/null || wget -q -O "$NCNN_ZIP" "$NCNN_URL" 2>/dev/null; then
            echo "[INFO] Extracting..."
            unzip -q -o "$NCNN_ZIP" -d "$SCRIPT_DIR/bin/"
            rm -f "$NCNN_ZIP"

            # Find and make executable
            FOUND_BIN=$(find "$SCRIPT_DIR/bin" -name "realesrgan-ncnn-vulkan" -type f 2>/dev/null | head -1)
            if [ -n "$FOUND_BIN" ]; then
                chmod +x "$FOUND_BIN"
                echo "[OK] realesrgan-ncnn-vulkan ready: $FOUND_BIN"
            else
                echo "[WARNING] Binary not found after extraction."
            fi
        else
            echo "[WARNING] Download failed. Upscale feature will not be available."
            rm -f "$NCNN_ZIP"
        fi
    fi
else
    echo "[OK] realesrgan-ncnn-vulkan is ready."
fi

echo ""
echo "[INFO] Starting Manga Translator..."
python app.py
