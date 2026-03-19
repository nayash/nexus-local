import mimetypes
import os
from dataclasses import dataclass


_KNOWN_EXTENSION_TO_SOURCE_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".log": "log",
    ".md": "md",
    ".csv": "csv",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".html": "html",
    ".htm": "html",
}

SUPPORTED_SOURCE_TYPES = {"pdf", "docx", "txt", "log", "md", "csv", "image", "html"}
SUPPORTED_EXTENSIONS = tuple(sorted(_KNOWN_EXTENSION_TO_SOURCE_TYPE.keys()))

_TEXTUAL_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-javascript",
)
_IMAGE_MIME_PREFIX = "image/"
_AUDIO_MIME_PREFIX = "audio/"
_VIDEO_MIME_PREFIX = "video/"
_BINARY_MIME_PREFIXES = (
    "application/zip",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/octet-stream",
    "application/pdf",
)
_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class DetectedFileType:
    source_type: str
    ingestible: bool
    detection_method: str
    mime_type: str = ""


def _looks_like_pdf(sample: bytes) -> bool:
    return sample.startswith(b"%PDF-")


def _looks_like_png(sample: bytes) -> bool:
    return sample.startswith(b"\x89PNG\r\n\x1a\n")


def _looks_like_jpeg(sample: bytes) -> bool:
    return sample.startswith(b"\xff\xd8\xff")


def _looks_like_zip_container(sample: bytes) -> bool:
    return sample.startswith(b"PK\x03\x04")


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\r\n\t")
    return printable / max(len(text), 1)


def _looks_like_delimited_text(text: str) -> bool:
    lines = [line for line in text.splitlines()[:12] if line.strip()]
    if len(lines) < 2:
        return False

    for delimiter in (",", "\t", ";", "|"):
        counts = [line.count(delimiter) for line in lines]
        if min(counts) <= 0:
            continue
        if max(counts) - min(counts) <= 1:
            return True
    return False


def _decode_text_sample(sample: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return sample.decode(encoding)
        except Exception:
            continue
    return ""


def detect_file_type(path: str) -> DetectedFileType:
    abs_path = os.path.abspath(os.path.expanduser(path))
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in _KNOWN_EXTENSION_TO_SOURCE_TYPE:
        source_type = _KNOWN_EXTENSION_TO_SOURCE_TYPE[ext]
        return DetectedFileType(source_type=source_type, ingestible=True, detection_method="extension")

    try:
        with open(abs_path, "rb") as handle:
            sample = handle.read(_SNIFF_BYTES)
    except OSError:
        return DetectedFileType(source_type="unknown", ingestible=False, detection_method="unreadable")

    if not sample:
        return DetectedFileType(source_type="unknown", ingestible=False, detection_method="empty")

    if _looks_like_pdf(sample):
        return DetectedFileType(source_type="pdf", ingestible=True, detection_method="signature", mime_type="application/pdf")
    if _looks_like_png(sample):
        return DetectedFileType(source_type="image", ingestible=True, detection_method="signature", mime_type="image/png")
    if _looks_like_jpeg(sample):
        return DetectedFileType(source_type="image", ingestible=True, detection_method="signature", mime_type="image/jpeg")
    if _looks_like_zip_container(sample) and ext != ".docx":
        return DetectedFileType(source_type="binary", ingestible=False, detection_method="signature", mime_type="application/zip")

    mime_type = mimetypes.guess_type(abs_path)[0] or ""
    if mime_type:
        if mime_type.startswith(_IMAGE_MIME_PREFIX):
            return DetectedFileType(source_type="image", ingestible=True, detection_method="mime", mime_type=mime_type)
        if mime_type.startswith(_AUDIO_MIME_PREFIX):
            return DetectedFileType(source_type="audio", ingestible=False, detection_method="mime", mime_type=mime_type)
        if mime_type.startswith(_VIDEO_MIME_PREFIX):
            return DetectedFileType(source_type="video", ingestible=False, detection_method="mime", mime_type=mime_type)
        if mime_type in _TEXTUAL_MIME_PREFIXES or mime_type.startswith("text/"):
            text_sample = _decode_text_sample(sample)
            if _looks_like_delimited_text(text_sample):
                return DetectedFileType(source_type="csv", ingestible=True, detection_method="mime+content", mime_type=mime_type)
            return DetectedFileType(source_type="txt", ingestible=True, detection_method="mime", mime_type=mime_type)
        if mime_type in _BINARY_MIME_PREFIXES:
            return DetectedFileType(source_type="binary", ingestible=False, detection_method="mime", mime_type=mime_type)

    if b"\x00" in sample:
        return DetectedFileType(source_type="binary", ingestible=False, detection_method="binary-sniff", mime_type=mime_type)

    text_sample = _decode_text_sample(sample)
    if text_sample:
        ratio = _printable_ratio(text_sample)
        if ratio >= 0.85:
            if _looks_like_delimited_text(text_sample):
                return DetectedFileType(source_type="csv", ingestible=True, detection_method="content-sniff", mime_type=mime_type)
            return DetectedFileType(source_type="txt", ingestible=True, detection_method="content-sniff", mime_type=mime_type)

    return DetectedFileType(source_type="unknown", ingestible=False, detection_method="unknown", mime_type=mime_type)
