from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2
import math
import os

# Font cache to avoid reloading fonts from disk
_font_cache = {}

# Font sizing configuration
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 80
PADDING_RATIO = 0.05  # 5% padding inside bubble

# Base directory for resolving relative font paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback font for Vietnamese (supports full Unicode diacritics)
_FALLBACK_FONT = os.path.join(_BASE_DIR, "fonts", "ariali.ttf")

# Cache for which font paths support Vietnamese
_viet_support_cache = {}


def _check_vietnamese_support(font_path):
    """
    Check if a font supports Vietnamese diacritical marks.
    
    Uses width variance: if all test Vietnamese characters have the exact same width,
    they're all mapping to the same .notdef glyph box → font doesn't support Vietnamese.
    Real fonts render đ, ạ, ơ, ữ, ẫ at different widths.
    """
    if font_path in _viet_support_cache:
        return _viet_support_cache[font_path]
    
    resolved_path = font_path
    if not os.path.isabs(font_path):
        resolved_path = os.path.join(_BASE_DIR, font_path)
    
    try:
        test_font = ImageFont.truetype(resolved_path, size=40)
        # These Vietnamese chars should have DIFFERENT widths in a supporting font
        # (đ is narrow, ơ/ư are wider, ẫ has diacritics affecting metrics)
        test_chars = ['đ', 'ạ', 'ơ', 'ữ', 'ẫ', 'ề']
        widths = set()
        for ch in test_chars:
            w = round(test_font.getlength(ch), 1)
            widths.add(w)
        
        # If all Vietnamese chars have the same width → all .notdef boxes
        has_support = len(widths) > 1
        
        _viet_support_cache[font_path] = has_support
        if not has_support:
            print(f"[FONT] '{os.path.basename(resolved_path)}' does not support Vietnamese → auto-fallback to ariali.ttf")
        else:
            print(f"[FONT] '{os.path.basename(resolved_path)}' supports Vietnamese ✓")
        return has_support
    except Exception as e:
        print(f"[FONT] Error checking '{font_path}': {e}")
        _viet_support_cache[font_path] = False
        return False


def _has_vietnamese(text):
    """Check if text contains Vietnamese diacritical characters."""
    viet_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
                     "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")
    return any(c in viet_chars for c in text)


def get_cached_font(font_path, size, text=None):
    """Get font from cache or load it. Auto-fallback for Vietnamese text if font doesn't support it."""
    # Check if we need Vietnamese fallback
    actual_font_path = font_path
    if text and _has_vietnamese(text):
        if not _check_vietnamese_support(font_path):
            actual_font_path = _FALLBACK_FONT
    
    cache_key = (actual_font_path, size)
    if cache_key not in _font_cache:
        # Resolve relative paths against the app's base directory
        resolved_path = actual_font_path
        if not os.path.isabs(actual_font_path):
            resolved_path = os.path.join(_BASE_DIR, actual_font_path)
        try:
            _font_cache[cache_key] = ImageFont.truetype(resolved_path, size=size)
        except Exception as e:
            print(f"[WARNING] Failed to load font '{resolved_path}': {e}")
            # Fallback to default font if custom font fails
            _font_cache[cache_key] = ImageFont.load_default()
    return _font_cache[cache_key]


def smart_wrap_text(text, chars_per_line):
    """
    Smart text wrapping that respects word boundaries.
    Avoids breaking Vietnamese words mid-character.
    
    Args:
        text: Text to wrap
        chars_per_line: Maximum characters per line
        
    Returns:
        Wrapped text with newlines
    """
    if not text or chars_per_line <= 0:
        return text
    
    # First try standard word-based wrapping (don't break words)
    wrapped = textwrap.fill(
        text, 
        width=chars_per_line, 
        break_long_words=False,  # Never break words mid-character!
        break_on_hyphens=False   # Don't break on hyphens
    )
    
    # If a single word is longer than the line, we need special handling
    lines = wrapped.split('\n')
    result_lines = []
    
    for line in lines:
        if len(line) <= chars_per_line:
            result_lines.append(line)
        else:
            # Line still too long (single long word) - break at space boundaries
            # For Vietnamese, try to break at spaces only
            words = line.split(' ')
            current_line = ""
            
            for word in words:
                if not current_line:
                    current_line = word
                elif len(current_line) + 1 + len(word) <= chars_per_line:
                    current_line += " " + word
                else:
                    if current_line:
                        result_lines.append(current_line)
                    current_line = word
            
            if current_line:
                result_lines.append(current_line)
    
    return '\n'.join(result_lines)


def calculate_optimal_font_size(text, w, h, font_path):
    """
    Calculate optimal font size to fill the bubble nicely.
    
    Args:
        text: Text to render
        w: Bubble width
        h: Bubble height
        font_path: Path to font file
        
    Returns:
        tuple: (font_size, line_height, wrapped_text, font)
    """
    # Apply padding
    usable_w = int(w * (1 - 2 * PADDING_RATIO))
    usable_h = int(h * (1 - 2 * PADDING_RATIO))
    
    if usable_w <= 0 or usable_h <= 0:
        return MIN_FONT_SIZE, MIN_FONT_SIZE, text, get_cached_font(font_path, MIN_FONT_SIZE, text=text)
    
    # Search from MAX down to MIN to find the LARGEST font size that fits
    best_font_size = MIN_FONT_SIZE
    best_wrapped = text
    found_fit = False
    
    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        font = get_cached_font(font_path, size, text=text)
        line_height = int(size * 1.3)
        
        # Calculate characters per line using actual font metrics
        # Use getlength for more accurate measurement
        try:
            avg_char_width = font.getlength("M")  # Use 'M' as reference (widest char)
        except:
            avg_char_width = size * 0.65
        chars_per_line = max(1, int(usable_w / avg_char_width))
        
        # Wrap text using smart wrapper (no word-breaking!)
        wrapped = smart_wrap_text(text, chars_per_line)
        lines = wrapped.split('\n')
        
        # Calculate total height needed
        total_height = len(lines) * line_height
        
        # Check if text fits vertically
        if total_height > usable_h:
            continue
        
        # Check if all lines fit width-wise
        fits_width = True
        for line in lines:
            try:
                line_width = font.getlength(line)
            except:
                line_width = len(line) * avg_char_width
            if line_width > usable_w:
                fits_width = False
                break
        
        if fits_width:
            best_font_size = size
            best_wrapped = wrapped
            found_fit = True
            break
    
    # If nothing fits even at MIN_FONT_SIZE, still wrap and truncate
    if not found_fit:
        font = get_cached_font(font_path, MIN_FONT_SIZE, text=text)
        line_height = int(MIN_FONT_SIZE * 1.3)
        try:
            avg_char_width = font.getlength("M")
        except:
            avg_char_width = MIN_FONT_SIZE * 0.65
        chars_per_line = max(1, int(usable_w / avg_char_width))
        wrapped = smart_wrap_text(text, chars_per_line)
        lines = wrapped.split('\n')
        
        # Truncate lines that don't fit vertically
        max_lines = max(1, usable_h // line_height)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            # Add ellipsis to last line
            if lines:
                last_line = lines[-1]
                if len(last_line) > 3:
                    lines[-1] = last_line[:-3] + "..."
                else:
                    lines[-1] = "..."
        
        best_wrapped = '\n'.join(lines)
        best_font_size = MIN_FONT_SIZE
    
    return best_font_size, int(best_font_size * 1.3), best_wrapped, get_cached_font(font_path, best_font_size, text=text)



def add_text(
    image,
    text,
    font_path,
    bubble_contour,
    text_color=(0, 0, 0),
    is_dark_bubble=False,
    detected_color=None,
    requires_stroke=False
):
    """
    Add text inside a speech bubble contour with dynamic font sizing.

    Args:
        image (numpy.ndarray): Processed bubble image (cv2 format - BGR).
        text (str): Text to be placed inside the speech bubble.
        font_path (str): Font path.
        bubble_contour (numpy.ndarray): Contour of the detected speech bubble.
        text_color (tuple): RGB color for text. Default is black (0,0,0).
                           Use (255,255,255) for white text on dark bubbles.

    Returns:
        numpy.ndarray: Image with text placed inside the speech bubble.
    """
    if not text or not text.strip():
        return image
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)
    
    # Calculate optimal font size
    font_size, line_height, wrapped_text, font = calculate_optimal_font_size(
        text, w, h, font_path
    )
    
    lines = wrapped_text.split('\n')
    total_text_height = len(lines) * line_height

    # Vertical centering - clamp to stay within bubble
    text_y = y + max(0, (h - total_text_height) // 2)

    img_h, img_w = image.shape[:2]
    
    for line in lines:
        try:
            text_length = font.getlength(line)
        except:
            text_length = len(line) * font_size * 0.6

        # Horizontal centering - clamp to stay within bubble
        text_x = x + max(0, (w - int(text_length)) // 2)
        
        # Don't render lines outside the image bounds
        if text_y >= img_h or text_y + line_height < 0:
            text_y += line_height
            continue

        if requires_stroke:
            # Use white stroke for black text, and black stroke for white text
            stroke_fill = "white" if text_color == (0, 0, 0) else "black"
            draw.text((text_x, text_y), line, font=font, fill=text_color, stroke_width=3, stroke_fill=stroke_fill)
        else:
            draw.text((text_x, text_y), line, font=font, fill=text_color)
            
        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image

