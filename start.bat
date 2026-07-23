@echo off
echo ===================================================
echo Manga Translator - Startup Script (Windows)
echo ===================================================

cd /d "%~dp0"

if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: =============================================================
:: System Dependencies (Tesseract OCR)
:: =============================================================
where tesseract >nul 2>&1
if errorlevel 1 (
    echo [INFO] Tesseract OCR not found.
    where choco >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Installing via Chocolatey...
        choco install tesseract -y
    ) else (
        echo [WARNING] Tesseract OCR not installed. Please install manually:
        echo   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        echo   2. During install, check "Additional language data" and select Chinese Traditional
        echo   3. Add Tesseract to PATH
        echo   Or install via Chocolatey: choco install tesseract
    )
)

:: =============================================================
:: Dependency Check - auto-detect and install missing packages
:: =============================================================
echo [INFO] Checking dependencies...

set MISSING=
set MISSING_COUNT=0

:: Check each critical package (import_name -> pip_name)
:: Manga-Translator deps
call :check_pkg flask Flask
call :check_pkg flask_socketio flask-socketio
call :check_pkg cv2 opencv-python
call :check_pkg PIL pillow
call :check_pkg torch torch
call :check_pkg torchvision torchvision
call :check_pkg numpy numpy
call :check_pkg tqdm tqdm
call :check_pkg deep_translator deep-translator
call :check_pkg translators translators
call :check_pkg manga_ocr manga-ocr
call :check_pkg ultralytics ultralytics
call :check_pkg safetensors safetensors
call :check_pkg cryptography cryptography
call :check_pkg sentencepiece sentencepiece
call :check_pkg werkzeug Werkzeug
call :check_pkg engineio python-engineio
call :check_pkg socketio python-socketio
call :check_pkg huggingface_hub huggingface-hub
call :check_pkg google.genai google-genai
call :check_pkg openai openai
call :check_pkg google.protobuf protobuf
call :check_pkg pytesseract pytesseract
call :check_pkg paddleocr paddleocr
:: PanelCleanerZ deps (used by pcleaner_bridge.py)
call :check_pkg pyclipper pyclipper
call :check_pkg shapely shapely
call :check_pkg scipy scipy
call :check_pkg loguru loguru
call :check_pkg packaging packaging

if %MISSING_COUNT%==0 (
    echo [OK] All dependencies are satisfied.
    goto :deps_done
)

echo.
echo [INFO] Installing %MISSING_COUNT% missing package(s):%MISSING%
pip install %MISSING%

:: Verify installation
set VERIFY_FAIL=0
call :verify_pkg flask Flask
call :verify_pkg flask_socketio flask-socketio
call :verify_pkg cv2 opencv-python
call :verify_pkg PIL pillow
call :verify_pkg torch torch
call :verify_pkg torchvision torchvision
call :verify_pkg numpy numpy
call :verify_pkg tqdm tqdm
call :verify_pkg deep_translator deep-translator
call :verify_pkg translators translators
call :verify_pkg manga_ocr manga-ocr
call :verify_pkg ultralytics ultralytics
call :verify_pkg safetensors safetensors
call :verify_pkg cryptography cryptography
call :verify_pkg sentencepiece sentencepiece
call :verify_pkg werkzeug Werkzeug
call :verify_pkg engineio python-engineio
call :verify_pkg socketio python-socketio
call :verify_pkg huggingface_hub huggingface-hub
call :verify_pkg google.genai google-genai
call :verify_pkg openai openai
call :verify_pkg google.protobuf protobuf
call :verify_pkg pytesseract pytesseract
call :verify_pkg paddleocr paddleocr
call :verify_pkg pyclipper pyclipper
call :verify_pkg shapely shapely
call :verify_pkg scipy scipy
call :verify_pkg loguru loguru
call :verify_pkg packaging packaging

if %VERIFY_FAIL% GTR 0 (
    echo [WARNING] Some packages could not be installed. Check errors above.
    echo [WARNING] You may need to install them manually.
    echo.
    set /p choice="Continue anyway? (y/N): "
    if /i not "%choice%"=="y" (
        echo Aborted.
        pause
        exit /b 1
    )
) else (
    echo [OK] All missing packages installed successfully!
)

:deps_done

:: Also sync with requirements file for any new additions
echo [INFO] Syncing with requirements-windows.txt...
pip install -q -r requirements-windows.txt 2>nul

echo.
echo [INFO] Starting Manga Translator...
python app.py

pause
exit /b 0

:: =============================================================
:: Subroutines
:: =============================================================

:check_pkg
:: %1 = import name, %2 = pip package name
python -c "import %~1" 2>nul
if errorlevel 1 (
    echo   [!] Missing: %~2 (%~1)
    set "MISSING=%MISSING% %~2"
    set /a MISSING_COUNT+=1
)
exit /b 0

:verify_pkg
:: %1 = import name, %2 = pip package name
python -c "import %~1" 2>nul
if errorlevel 1 (
    set /a VERIFY_FAIL+=1
)
exit /b 0
