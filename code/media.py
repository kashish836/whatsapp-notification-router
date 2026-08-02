"""
media.py  (Groq edition — 100% free, zero local ML models)
--------------------------------------------------------------
Groq provides:
  - Vision via chat completions with image_url content blocks (base64 data URI)
  - Audio transcription via a dedicated /audio/transcriptions endpoint (Whisper)

Both functions cache results to disk (code/.media_cache/) so re-runs
during development don't re-pay API quota for the same file.
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import mimetypes
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".media_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(file_path: str, kind: str) -> str:
    h = hashlib.sha1(file_path.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{kind}_{h}.json")


def _guess_mime(file_path: str, fallback: str) -> str:
    mt, _ = mimetypes.guess_type(file_path)
    return mt or fallback


def describe_image(client, model: str, file_path: str) -> str:
    """Returns OCR'd text + a short red-flag-aware description of the image."""
    cache = _cache_path(file_path, "img")
    if os.path.exists(cache):
        return json.load(open(cache))["text"]

    if not os.path.exists(file_path):
        return "[image file not found]"

    with open(file_path, "rb") as f:
        img_bytes = f.read()
    mime = _guess_mime(file_path, "image/jpeg")
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "This is a WhatsApp image message (poster, screenshot, or photo). "
        "Do two things concisely:\n"
        "1) Transcribe any visible text verbatim (OCR).\n"
        "2) In one short line, describe what kind of content this is "
        "(e.g. sale poster, payment/QR screenshot, event flyer, meme, "
        "official notice, phishing-style urgency banner) and flag anything "
        "that looks manipulative, scammy, or urgency-baiting.\n"
        "Keep the whole answer under 120 words."
    )
    text = "[image analysis failed]"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            time.sleep(1.5)
            break
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries - 1:
                time.sleep(5.0 * (attempt + 1))
                continue
            text = f"[image analysis failed: {e}]"

    json.dump({"text": text}, open(cache, "w"))
    return text


def transcribe_audio(client, model: str, file_path: str) -> str:
    """Groq Whisper transcription via the dedicated audio endpoint."""
    cache = _cache_path(file_path, "audio")
    if os.path.exists(cache):
        return json.load(open(cache))["text"]

    if not os.path.exists(file_path):
        return "[voice note file not found]"

    text = "[transcription unavailable]"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=(os.path.basename(file_path), f),
                )
            text = (resp.text or "").strip() or "[transcription empty / silence]"
            time.sleep(1.5)
            break
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries - 1:
                time.sleep(5.0 * (attempt + 1))
                continue
            text = f"[audio transcription unavailable: {e}]"

    json.dump({"text": text}, open(cache, "w"))
    return text


def get_media_text(client, model: str, media_type: str, file_path: str | None) -> str:
    if not file_path:
        return ""
    if media_type == "image":
        return describe_image(client, model, file_path)
    if media_type == "voice":
        return transcribe_audio(client, model, file_path)
    return ""
