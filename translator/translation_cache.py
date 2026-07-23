from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import threading
import time
from typing import Any, Dict, Optional


class TranslationCache:
    """
    Disk-based cache for OCR and translation results.
    Caches results indexed by MD5 hash of image bytes and translation settings.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(project_root, "cache", "translations")

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cleanup()

    def _make_key(
        self,
        image_bytes: bytes,
        ocr_engine: str,
        source_lang: str,
        translator_type: str,
        target_lang: str,
        style: str = ''
    ) -> str:
        """
        Generate MD5 hex digest key for the given image bytes and settings.
        """
        hasher = hashlib.md5()
        if isinstance(image_bytes, bytes):
            hasher.update(image_bytes)
        else:
            hasher.update(str(image_bytes).encode('utf-8'))

        for param in (ocr_engine, source_lang, translator_type, target_lang, style):
            hasher.update(f":{param}".encode('utf-8'))

        return hasher.hexdigest()

    def get(
        self,
        image_bytes: bytes,
        ocr_engine: str,
        source_lang: str,
        translator_type: str,
        target_lang: str,
        style: str = ''
    ) -> Dict[str, Any] | None:
        """
        Retrieve cached translation data if available.
        """
        key = self._make_key(image_bytes, ocr_engine, source_lang, translator_type, target_lang, style)
        filepath = os.path.join(self.cache_dir, f"{key}.json")

        if not os.path.isfile(filepath):
            return None

        with self._lock:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return None

    def put(
        self,
        image_bytes: bytes,
        ocr_engine: str,
        source_lang: str,
        translator_type: str,
        target_lang: str,
        style: str,
        data: dict
    ) -> None:
        """
        Store translation data into cache.
        """
        key = self._make_key(image_bytes, ocr_engine, source_lang, translator_type, target_lang, style)
        filepath = os.path.join(self.cache_dir, f"{key}.json")

        entry = {
            "ocr_texts": data.get("ocr_texts", []),
            "translated_texts": data.get("translated_texts", []),
            "bubbles_data": data.get("bubbles_data", []),
            "timestamp": data.get("timestamp", time.time()),
            "settings": {
                "ocr": ocr_engine,
                "source": source_lang,
                "target": target_lang,
                "translator": translator_type,
                "style": style
            }
        }

        # Preserve any additional fields provided in data
        for k, v in data.items():
            if k not in entry:
                entry[k] = v

        with self._lock:
            tmp_path = filepath + f".tmp.{threading.get_ident()}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, filepath)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def clear(self) -> None:
        """
        Delete all cache files in the cache directory.
        """
        with self._lock:
            pattern = os.path.join(self.cache_dir, "*.json")
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                except OSError:
                    pass

            tmp_pattern = os.path.join(self.cache_dir, "*.tmp*")
            for filepath in glob.glob(tmp_pattern):
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    def cleanup(self, max_age_days: int = 7) -> int:
        """
        Delete entries older than max_age_days.
        Returns count of removed files.
        """
        cutoff = time.time() - (max_age_days * 86400)
        removed_count = 0
        with self._lock:
            pattern = os.path.join(self.cache_dir, "*.json")
            for filepath in glob.glob(pattern):
                try:
                    st_mtime = os.path.getmtime(filepath)
                    should_delete = False
                    if st_mtime < cutoff:
                        should_delete = True
                    else:
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                content = json.load(f)
                                ts = content.get("timestamp", st_mtime)
                                if ts < cutoff:
                                    should_delete = True
                        except (OSError, json.JSONDecodeError):
                            should_delete = True

                    if should_delete:
                        os.remove(filepath)
                        removed_count += 1
                except OSError:
                    pass

            # Also clean up lingering temp files
            tmp_pattern = os.path.join(self.cache_dir, "*.tmp*")
            for filepath in glob.glob(tmp_pattern):
                try:
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                except OSError:
                    pass

        return removed_count

    def stats(self) -> Dict[str, Any]:
        """
        Return dictionary with cache statistics: {"count": int, "size_mb": float}
        """
        count = 0
        total_bytes = 0
        with self._lock:
            pattern = os.path.join(self.cache_dir, "*.json")
            for filepath in glob.glob(pattern):
                try:
                    total_bytes += os.path.getsize(filepath)
                    count += 1
                except OSError:
                    pass

        size_mb = round(total_bytes / (1024 * 1024), 2)
        return {
            "count": count,
            "size_mb": size_mb
        }


_cache_instance = None


def get_cache() -> TranslationCache:
    """
    Module-level convenience function returning a shared TranslationCache singleton instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TranslationCache()
    return _cache_instance
