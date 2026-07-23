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

echo ""
echo "[INFO] Starting Manga Translator..."
python app.py
