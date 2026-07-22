---
title: Manga Translator
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Manga Translator 📚

Dịch tự động speech bubbles trong manga/manhwa/manhua với AI!

🌐 **Demo:** [manga-translator.duongkum999.me](https://manga-translator.duongkum999.me)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **YOLO Detection** | Phát hiện speech bubble tự động (kể cả bubble đen) |
| 📝 **OCR** | Manga-OCR, Chrome Lens |
| 🌐 **Translators** | Gemini, Local LLM (Ollama/LM Studio), NLLB |
| 🧠 **Context Memory** | Dịch chính xác hơn với context từ nhiều trang |
| 🎨 **24+ Fonts** | Auto font matching với Gemini Vision |
| 📦 **Download ZIP** | Tải tất cả ảnh đã dịch |

## � Quick Start

```bash
# Clone
git clone https://github.com/pedguedes090/Manga-Translator.git
cd Manga-Translator

# On Linux/Mac
pip install -r requirements.txt

# On Windows
pip install -r requirements-windows.txt

# Run
python app.py
```

Mở http://localhost:5000

## � Translators

### Gemini (Recommended)
- Lấy API key từ [aistudio.google.com](https://aistudio.google.com/)
- Free tier: 15 RPM, 1M tokens/day

### Local LLM (Ollama / LM Studio)
- Chạy Ollama: `ollama serve` (port 11434)
- Hoặc LM Studio Server (port 1234)
- Nhập tên model: `llama3.2`, `qwen2.5`, `mistral`...

## �📋 Workflow

1. **Upload** manga/manhwa images
2. **Chọn ngôn ngữ** (Japanese/Chinese/Korean → Vietnamese/English/...)
3. **Chọn translator** (Gemini hoặc Local LLM)
4. **Enable Context Memory** để dịch chính xác hơn
5. **Click Translate** và xem progress real-time
6. **Download** từng ảnh hoặc ZIP

## 🌍 Languages

**Source:** Japanese, Chinese, Korean, English  
**Target:** Vietnamese, English, Chinese, Korean, Thai, Indonesian, French, German, Spanish, Russian

##  Tech Stack

- **Backend:** Flask + Flask-SocketIO
- **Detection:** YOLOv8 + OpenCV (black bubbles)
- **OCR:** Manga-OCR, Chrome Lens API
- **Translation:** Gemini API, OpenAI-compatible endpoints
- **Rendering:** PIL with smart text wrapping

## 📦 Docker

```bash
docker build -t manga-translator .
docker run -p 5000:5000 manga-translator
```

## 📄 License

MIT
