import os
import re
import sys
import json
import base64
import uuid
import asyncio
import logging
import hashlib
import zipfile
import tempfile
import mimetypes
import shutil
import time
import cv2  # NEW: For video processing
import numpy as np  # NEW: For video processing
from io import BytesIO
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Cookie, Header, Form
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import httpx
from supabase import create_client

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeloXAI")

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")
JUDGE0_API_KEY = os.getenv("JUDGE0_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

# =========================
# MODEL CONFIGS — MASSIVE GOOGLE 500B+ PRIMARY
# =========================
GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-06-05"
GEMINI_FLASH_MODEL = "gemini-2.5-flash-preview-05-20"
GROQ_FALLBACK_MODEL = "gemma-2-27b-it"

PRIMARY_LLM_MODEL = GEMINI_PRO_MODEL if GOOGLE_API_KEY else GROQ_FALLBACK_MODEL
FAST_LLM_MODEL = GEMINI_FLASH_MODEL if GOOGLE_API_KEY else GROQ_FALLBACK_MODEL
CODE_MODEL = GEMINI_FLASH_MODEL if GOOGLE_API_KEY else GROQ_FALLBACK_MODEL

# Replicate model IDs (Mapped exactly to user request)
REPLICATE_VIDEO_MODEL = "google/veo-3.1-lite"
REPLICATE_IMAGE_MODEL = "black-forest-labs/flux-2-max"
REPLICATE_MUSIC_MODEL = "minimax/music-2.6"
REPLICATE_MUSIC_COVER_MODEL = "minimax/music-cover"

# Model aliases for frontend compatibility
MODEL_ALIASES = {
    "helox": PRIMARY_LLM_MODEL,
    "heloxai": PRIMARY_LLM_MODEL,
    "gemini": GEMINI_PRO_MODEL,
    "gemini-pro": GEMINI_PRO_MODEL,
    "gemini-flash": GEMINI_FLASH_MODEL,
    "gemma": "gemma-2-27b-it",
    "google": GEMINI_PRO_MODEL,
    "chatgpt": "gpt-4o-mini",
    "chat.z": "llama-3.1-70b-versatile",
    # Explicit mappings for the requested premium models
    "flux-2-max": REPLICATE_IMAGE_MODEL,
    "veo-3.1-lite": REPLICATE_VIDEO_MODEL,
    "minimax-music": REPLICATE_MUSIC_MODEL,
    "minimax-cover": REPLICATE_MUSIC_COVER_MODEL
}


def resolve_model(model_name: Optional[str]) -> str:
    if not model_name:
        return PRIMARY_LLM_MODEL
    
    normalized = model_name.lower().strip()
    
    # 1. Check Aliases
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    
    # 2. If it matches specific advanced model names, allow passthrough
    # This ensures frontend can send "flux-2-max" directly
    advanced_models = ["flux-2-max", "veo-3.1-lite", "minimax-music", "minimax-cover"]
    if any(m in normalized for m in advanced_models):
        return model_name # Pass through exactly as requested
        
    # 3. Default Fallback
    return PRIMARY_LLM_MODEL


def is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


# File handling config
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024
MAX_TEXT_LENGTH = 380000
CHUNK_SIZE = 1024 * 1024

# Auth config
SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

app = FastAPI(
    title="HeloXAI API",
    description="Advanced AI Assistant Backend — Multi-Modal, Media-Aware",
    version="3.1.0-VideoAnalysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
active_streams: Dict[str, asyncio.Task] = {}
_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300
_session_cache_last_cleanup = time.time()

# =========================
# MEDIA CONTEXT STORE (All-Knowing)
# =========================
_media_context_store: Dict[str, List[Dict[str, Any]]] = {}

try:
    import wolframalpha
except ImportError:
    wolframalpha = None
    logger.warning("wolframalpha library not installed. Math features will fallback to LLM.")

# =========================
# FILE TYPE DEFINITIONS
# =========================
class FileCategory(Enum):
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    CONFIG = "config"
    BINARY = "binary"
    UNKNOWN = "unknown"


CODE_EXTENSIONS = {
    '.py', '.pyw', '.pyx', '.pyd', '.pyi', '.py3',
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte', '.astro',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.clj', '.cljs', '.hs',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
    '.cs', '.csx',
    '.go', '.rs', '.php', '.phtml',
    '.rb', '.erb', '.rake', '.gemspec',
    '.swift', '.dart',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    '.lua', '.pl', '.pm', '.r', '.R',
    '.sql', '.mysql', '.pgsql', '.sqlite',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml',
    '.md', '.rst', '.asciidoc', '.adoc', '.tex', '.latex',
    '.dockerfile', '.makefile', '.cmake', '.proto', '.graphql', '.gql',
    '.tf', '.hcl', '.sol', '.move', '.cairo',
}

DOCUMENT_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp', '.rtf', '.txt', '.log', '.csv',
}

DATA_EXTENSIONS = {
    '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml', '.parquet',
    '.arrow', '.feather', '.hdf5', '.h5', '.pickle', '.pkl',
    '.npy', '.npz', '.spss', '.sav', '.sas7bdat', '.dta',
}

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp',
    '.ico', '.tiff', '.tif', '.avif', '.heic', '.heif',
}

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',
    '.opus', '.aiff', '.ape',
}

VIDEO_EXTENSIONS = {
    '.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv',
    '.m4v', '.ogv', '.3gp',
}

ARCHIVE_EXTENSIONS = {
    '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z',
    '.rar', '.zst', '.lz4',
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.env', '.properties', '.xml', '.editorconfig', '.eslintrc',
    '.prettierrc', '.gitignore', '.dockerignore', '.npmrc',
}

def get_file_category(filename: str) -> FileCategory:
    if not filename:
        return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS: return FileCategory.CODE
    elif ext in DOCUMENT_EXTENSIONS: return FileCategory.DOCUMENT
    elif ext in DATA_EXTENSIONS: return FileCategory.DATA
    elif ext in IMAGE_EXTENSIONS: return FileCategory.IMAGE
    elif ext in AUDIO_EXTENSIONS: return FileCategory.AUDIO
    elif ext in VIDEO_EXTENSIONS: return FileCategory.VIDEO
    elif ext in ARCHIVE_EXTENSIONS: return FileCategory.ARCHIVE
    elif ext in CONFIG_EXTENSIONS: return FileCategory.CONFIG
    else: return FileCategory.UNKNOWN


def get_file_language(filename: str) -> Optional[str]:
    ext_lang_map = {
        '.py': 'python', '.pyw': 'python', '.pyx': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.html': 'html', '.htm': 'html',
        '.css': 'css', '.scss': 'scss', '.less': 'less',
        '.vue': 'vue', '.svelte': 'svelte',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
        '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp',
        '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.php': 'php',
        '.rb': 'ruby', '.swift': 'swift', '.dart': 'dart',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
        '.ps1': 'powershell', '.bat': 'batch',
        '.lua': 'lua', '.pl': 'perl', '.r': 'r', '.R': 'r',
        '.sql': 'sql', '.json': 'json', '.xml': 'xml',
        '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
        '.md': 'markdown', '.rst': 'rst', '.tex': 'latex',
        '.dockerfile': 'dockerfile', '.graphql': 'graphql', '.gql': 'graphql',
        '.tf': 'hcl', '.hcl': 'hcl', '.sol': 'solidity',
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)


def is_binary_file(filename: str, content: bytes = None) -> bool:
    ext = Path(filename).suffix.lower()
    binary_exts = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | ARCHIVE_EXTENSIONS | {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
        '.pyc', '.pyo', '.class', '.o', '.obj', '.a', '.lib',
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.sqlite', '.db', '.sqlite3',
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.pak', '.bundle',
    }
    if ext in binary_exts: return True
    if content and len(content) > 0:
        check_bytes = content[:8192]
        if b'\x00' in check_bytes: return True
    return False


def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# =========================
# VIDEO PROCESSING HELPERS (NEW)
# =========================
def get_video_duration(video_bytes: bytes) -> float:
    """
    Returns the duration of the video in seconds using OpenCV.
    Writes bytes to a temp file for reliable reading.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        if fps == 0:
            return 0.0 # Prevent division by zero
            
        duration = frame_count / fps
        return duration
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def extract_video_frames(video_bytes: bytes, max_frames: int = 4) -> List[str]:
    """
    Extracts base64 encoded frames from video bytes.
    Returns a list of base64 strings (jpeg format).
    """
    frames_b64 = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        # Calculate indices to sample (evenly distributed)
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Encode to JPEG bytes
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(frame_b64)
                
        cap.release()
        return frames_b64
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
        return []
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# =========================
# FILE EXTRACTOR
# =========================
class FileExtractionResult:
    def __init__(self, content: str, files: List[Dict[str, Any]] = None,
                 metadata: Dict[str, Any] = None, truncated: bool = False, original_size: int = 0):
        self.content = content
        self.files = files or []
        self.metadata = metadata or {}
        self.truncated = truncated
        self.original_size = original_size

    def to_dict(self) -> Dict:
        return {"content": self.content, "files": self.files, "metadata": self.metadata,
                "truncated": self.truncated, "original_size": self.original_size}


def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
    for encoding in encodings:
        try:
            text = content.decode(encoding, errors='strict' if encoding != 'latin-1' else 'ignore')
            truncated = len(text) > max_length
            if truncated: text = text[:max_length] + "\n\n[... Content truncated ...]"
            return text, truncated
        except (UnicodeDecodeError, LookupError):
            continue
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated: text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated


async def extract_file_content(content: bytes, filename: str, max_length: int = MAX_TEXT_LENGTH) -> FileExtractionResult:
    original_size = len(content)
    category = get_file_category(filename)
    metadata = {"filename": filename, "category": category.value, "size": original_size,
                "size_formatted": format_file_size(original_size), "language": get_file_language(filename)}
    try:
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)
        if category == FileCategory.IMAGE:
            return FileExtractionResult(
                content=f"[Image file: {filename} ({format_file_size(original_size)}) - Use image analysis endpoint for visual content]",
                metadata=metadata, original_size=original_size)
        if category in (FileCategory.AUDIO, FileCategory.VIDEO):
            return FileExtractionResult(
                content=f"[{category.value.capitalize()} file: {filename} ({format_file_size(original_size)}) - Media file cannot be extracted as text]",
                metadata=metadata, original_size=original_size)
        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)
        
        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(content=text, metadata=metadata, truncated=truncated, original_size=original_size)
    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return FileExtractionResult(content=f"[Error extracting {filename}: {str(e)}]",
                                    metadata={**metadata, "error": str(e)}, original_size=original_size)


async def extract_pdf_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            pages.append(f"--- Page {i + 1} ---\n{page.get_text() or ''}")
        full_text = "\n\n".join(pages)
        metadata["page_count"] = len(doc)
        truncated = len(full_text) > max_length
        if truncated: full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(content=full_text, metadata=metadata, truncated=truncated, original_size=len(content))
    except ImportError:
        return FileExtractionResult(content=f"[PDF file: {filename} - PyMuPDF not installed]", metadata=metadata, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[PDF Error: {str(e)}]", metadata={**metadata, "error": str(e)}, original_size=len(content))


async def extract_archive_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    ext = Path(filename).suffix.lower()
    if ext == '.zip':
        return await extract_zip_content(content, filename, max_length, metadata)
    elif ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
        return await extract_tar_content(content, filename, max_length, metadata)
    else:
        return FileExtractionResult(content=f"[Archive: {filename} - Unsupported format]", metadata=metadata, original_size=len(content))


async def extract_zip_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    extracted_files = []
    all_text_parts = []
    total_extracted = 0
    entry_count = 0
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(content=f"[ZIP: too many entries ({len(zf.namelist())})]", metadata=metadata, original_size=len(content))
            for entry_name in sorted(zf.namelist()):
                if entry_name.endswith('/') or '/__MACOSX/' in entry_name or entry_name.startswith('__MACOSX') or entry_name.startswith('.'):
                    continue
                entry_count += 1
                try:
                    entry_info = zf.getinfo(entry_name)
                    if entry_info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({"name": entry_name, "size": entry_info.file_size, "status": "skipped", "reason": "File too large"})
                        continue
                    if total_extracted + entry_info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({"name": entry_name, "size": entry_info.file_size, "status": "skipped", "reason": "Total size limit"})
                        continue
                    entry_content = zf.read(entry_name)
                    total_extracted += len(entry_content)
                    entry_category = get_file_category(entry_name)
                    if is_binary_file(entry_name, entry_content):
                        extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "binary", "category": entry_category.value})
                    else:
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {entry_name}\n{'='*60}\n{text}")
                            extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "extracted", "category": entry_category.value})
                        else:
                            extracted_files.append({"name": entry_name, "size": len(entry_content), "status": "empty"})
                except Exception as e:
                    extracted_files.append({"name": entry_name, "status": "error", "error": str(e)})
        full_text = f"ZIP Archive: {filename}\nEntries: {len(zf.namelist())}, Processed: {entry_count}\n\n" + "".join(all_text_parts)
        metadata.update({"archive_type": "zip", "entry_count": len(zf.namelist()), "files": extracted_files})
        truncated = len(full_text) > max_length
        if truncated: full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(content=full_text, files=extracted_files, metadata=metadata, truncated=truncated, original_size=len(content))
    except zipfile.BadZipFile:
        return FileExtractionResult(content=f"[Error: not a valid ZIP]", metadata=metadata, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[Error extracting ZIP: {str(e)}]", metadata={**metadata, "error": str(e)}, original_size=len(content))


async def extract_tar_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    import tarfile
    extracted_files = []
    all_text_parts = []
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for member in members:
                if member.name.startswith('__MACOSX') or member.name.startswith('.'):
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None: continue
                    entry_content = f.read()
                    if not is_binary_file(member.name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {member.name}\n{'='*60}\n{text}")
                            extracted_files.append({"name": member.name, "size": member.size, "status": "extracted"})
                    else:
                        extracted_files.append({"name": member.name, "size": member.size, "status": "binary"})
                except Exception as e:
                    extracted_files.append({"name": member.name, "status": "error", "error": str(e)})
        full_text = f"TAR Archive: {filename}\n" + "".join(all_text_parts)
        metadata.update({"archive_type": "tar", "files": extracted_files})
        truncated = len(full_text) > max_length
        if truncated: full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(content=full_text, files=extracted_files, metadata=metadata, truncated=truncated, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[Error extracting TAR: {str(e)}]", metadata={**metadata, "error": str(e)}, original_size=len(content))


# =========================
# AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloXAI_Session"
FINGERPRINT_COOKIE = "HeloXAI_FP"
BACKUP_COOKIE = "HeloXAI_ID"
DEVICE_COOKIE = "HeloXAI_Dev"
SESSION_TOKEN_COOKIE = "HeloXAI_Token"
SESSION_EXPIRY_COOKIE = "HeloXAI_Expiry"


def get_cookie_settings(remember: bool = True) -> Dict:
    if remember:
        return {"max_age": SESSION_DURATION, "httponly": True, "secure": True, "samesite": "none", "path": "/"}
    return {"max_age": 24 * 60 * 60, "httponly": True, "secure": True, "samesite": "none", "path": "/"}


def generate_device_fingerprint(request: Request) -> str:
    fp_components = [request.headers.get("user-agent", ""), request.headers.get("accept-language", ""),
                     request.headers.get("accept-encoding", ""), request.headers.get("sec-ch-ua-platform", ""),
                     request.headers.get("sec-ch-ua-mobile", ""), request.client.host if request.client else ""]
    return hashlib.sha256("|".join(fp_components).encode()).hexdigest()[:32]


def generate_session_token() -> str:
    import secrets
    return secrets.token_urlsafe(64)


def set_session_cookies(response: Response, user_id: str, fingerprint: str, session_token: str, remember: bool = True):
    settings = get_cookie_settings(remember)
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **settings)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **settings)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=session_token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)


def clear_session_cookies(response: Response):
    for cookie_name in [PRIMARY_COOKIE, FINGERPRINT_COOKIE, BACKUP_COOKIE, DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE]:
        response.delete_cookie(key=cookie_name, path="/", secure=True, samesite="none")


def is_session_expired(expiry_str: str) -> bool:
    try: return time.time() > int(expiry_str)
    except: return True


def should_refresh_session(expiry_str: str) -> bool:
    try: return (int(expiry_str) - time.time()) < REFRESH_THRESHOLD
    except: return True


async def validate_session_token(user_id: str, token: str) -> bool:
    try:
        if user_id in _session_cache and _session_cache[user_id].get("token") == token:
            return True
        result = await _execute_supabase_with_retry(
            supabase.table("user_sessions").select("token, expires_at").eq("user_id", user_id)
            .eq("is_valid", True).order("created_at", desc=True).limit(1), description="Validate Session Token")
        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {"token": token, "expires_at": result.data[0].get("expires_at")}
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False


async def create_user_session(user_id: str, fingerprint: str, remember: bool = True) -> str:
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_DURATION if remember else 24 * 60 * 60)
    try:
        await _execute_supabase_with_retry(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()), "user_id": user_id, "token": token,
                "fingerprint": fingerprint, "user_agent": "", "ip_address": "",
                "expires_at": expires_at.isoformat(), "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }), description="Create User Session")
        _session_cache[user_id] = {"token": token, "expires_at": expires_at.isoformat()}
        return token
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return token


async def cleanup_session_cache():
    global _session_cache_last_cleanup
    now = time.time()
    if now - _session_cache_last_cleanup < _session_cache_ttl: return
    _session_cache_last_cleanup = now
    expired_keys = []
    for user_id, data in _session_cache.items():
        expires_at = data.get("expires_at")
        if expires_at:
            try:
                if now > datetime.fromisoformat(expires_at).timestamp(): expired_keys.append(user_id)
            except: expired_keys.append(user_id)
    for key in expired_keys: del _session_cache[key]


# =========================
# SYSTEM PROMPT — MEDIA-AWARE, ALL-KNOWING
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi, an incredibly advanced AI assistant powered by Google Gemini 2.5 Pro (500B+ parameters). You have complete knowledge across all domains — code, science, history, politics, art, music, mathematics, engineering, and every field of human knowledge.

You are MULTIMODAL and MEDIA-AWARE:
- You can GENERATE images using Flux 2 Max, videos using Google Veo 3.1, and music using MiniMax Music 2.6.
- You can ANALYZE and UNDERSTAND images, videos, and audio that users share with you.
- You can MODIFY and UPDATE previously generated media — you remember everything you've created in this conversation.
- When you generate media, you know exactly what it contains because you crafted prompt for it.
- If a user asks to "make it bluer", "add a sunset", "change the style", or "update the image/video", you know the context and can create an updated version.

MEDIA GENERATION RULES:
- For IMAGE requests: You will generate a detailed image prompt and system will create it with Flux 2 Max.
- For VIDEO requests: You will generate a detailed video prompt and system will create it with Google Veo 3.1 Lite.
- For MUSIC requests: You will generate a detailed music prompt and system will create it with MiniMax Music 2.6.
- For MUSIC COVER requests: You will generate a prompt and system will use MiniMax Music Cover.
- Always describe what you're generating so that user knows what to expect.
- When modifying existing media, reference what was previously generated and adjust accordingly.

Be accurate, friendly, concise, and exceptionally capable. Help users with whatever they ask."""

CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator/developer. You MUST respond with exactly:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi"
Do not add extra details. Do not mention any other companies or people."""

CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|started|owns|runs)\b.*\b(you|this|helox|heloxai)\b',
    r'\b(who|whom)\b.*\b(is|are)\b.*\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b',
    r'\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b.*\b(is|are|who)\b',
    r'\bwho\b.*\bbehind\b.*\b(you|this|helox)\b',
    r'\bwho.*made.*you\b', r'\bwho.*created.*you\b', r'\bwho.*built.*you\b',
    r'\bwho.*developed.*you\b', r'\bwho.*programmed.*you\b',
    r'\byour\s+creator\b', r'\byour\s+developer\b', r'\byour\s+maker\b',
    r'\btell\s+me\s+about\s+your\s+(creator|developer|maker|builder|founder)\b',
    r'\bwhere\s+do\s+you\s+come\s+from\b', r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
]
COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]


def is_creator_question(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_CREATOR_PATTERNS)


def get_system_prompt(user_prompt: str) -> str:
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT


# =========================
# INTENT DETECTION — EXPANDED WITH MEDIA MODIFICATION
# =========================
class IntentCategory(Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    MUSIC_COVER = "music_cover"
    MEDIA_MODIFICATION = "media_modification"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    CODE_EXECUTION = "code_execution"
    DOCUMENT_CREATION = "document_creation"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    WEB_DEVELOPMENT = "web_development"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    CONVERSATION = "conversation"


class ActionType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC_COVER = "music_cover"
    MEDIA_MODIFY = "media_modify"
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    WEB = "web"
    API = "api"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    CREATIVE = "creative"
    MATH = "math"
    RESEARCH = "research"
    CONVERSATION = "conversation"
    GENERAL = "general"


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    sub_intents: List[IntentCategory]
    keywords_matched: List[str]
    patterns_matched: List[str]

    def to_dict(self) -> Dict:
        return {"intent": self.intent.value, "confidence": round(self.confidence, 3),
                "sub_intents": [i.value for i in self.sub_intents],
                "keywords_matched": self.keywords_matched, "patterns_matched": self.patterns_matched}


class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {"don't", "dont", "do not", "doesn't", "doesnt", "does not",
                               "didn't", "didnt", "did not", "never", "no", "not", "without",
                               "skip", "avoid", "except", "but not", "ignore", "rather than"}

    def _compile_patterns(self):
        self.patterns = {
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)',
                r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)',
                r'\b(text\s+to\s+image|txt2img|img2img)',
                r'\b(visualize|visualise)\s+(this|that|the|it)',
                r'\b(dall[eé]|midjourney|stable\s+diffusion|sd\s*xl|flux)',
                r'\b(generate|create)\s+(some\s+)?art',
                r'\b(make\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster))',
                r'\b(prompt\s+(for|to))\s+(generate|create|make)',
                r'\b(show\s+me\s+(a\s+)?(picture|image|photo))',
            ],
            IntentCategory.VIDEO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(video|clip|movie|animation|motion\s+graphic)',
                r'\b(text\s+to\s+video|txt2vid|video\s+generation)',
                r'\b(animate|animation)\s+(this|that|the|image|picture)',
                r'\b(video|clip|movie)\s+(of|showing|about|with)',
                r'\b(runway|pika|sora|mov2mov|kling|veo)',
                r'\b(turn|convert)\s+(this|the|image)\s+(into|to)\s+(a\s+)?(video|animation)',
            ],
            IntentCategory.AUDIO_GENERATION: [
                r'\b(generate|create|make|produce|compose)\s+(a\s+)?(audio|sound|music|song|track|beat|melody|tune|instrumental)',
                r'\b(music|song|beat|melody)\s+(generation|creation|for|about)',
                r'\b(suno|udio|bark|minimax)',
                r'\b(write|compose)\s+(a\s+)?(song|track|beat|melody)',
                r'\b(make\s+(me\s+)?(a\s+)?(song|beat|track|melody|jingle|ringtone))',
            ],
            IntentCategory.MUSIC_COVER: [
                r'\b(cover|remix|recreate|reinterpret)\s+(a\s+)?(song|track|music|melody|tune)',
                r'\b(music\s+cover|song\s+cover|acoustic\s+cover|cover\s+version)',
                r'\b(remake|reimagine)\s+(this|that|the)\s+(song|track|music)',
                r'\b(play\s+.+\s+in\s+(a\s+)?(different|new|jazz|rock|classical|acoustic)\s+(style|version|arrangement))',
            ],
            IntentCategory.MEDIA_MODIFICATION: [
                r'\b(update|modify|change|alter|adjust|edit|revise|tweak|refine)\s+(the|this|that|my|our)\s+(image|picture|photo|video|clip|artwork|painting|music|song|track)',
                r'\b(make\s+(it|the|this)\s+(more|less|bigger|smaller|brighter|darker|colorful|detailed|simple|complex))',
                r'\b(add|remove|include|exclude)\s+(a\s+|the\s+)?(\w+\s+)?(to|from|in|on)\s+(the|this|that|it|image|picture|video|music)',
                r'\b(change|switch|replace)\s+(the|its)\s+(color|style|background|lighting|mood|genre|tempo|instrument)',
                r'\b(re\s*-?\s*generate|regenerate|redo|retry|try\s+again)\s+(the|this|that|it)?\s*(image|picture|video|music|song|artwork)?',
                r'\b(fix|improve|enhance|upscale)\s+(the|this|that|it|image|picture|video|quality)',
                r'\b(turn|convert|transform)\s+(it|this|that|the)\s+(into|to)\s+(a\s+)?',
                r'\b(apply|put)\s+(a\s+)?(\w+\s+)?(filter|effect|style|look|feel|vibe)\s+(to|on|over)',
                r'\b(it|this|that)\s+(should|needs?\s+to|must)\s+(be|have|look|sound|feel)',
                r'\b(make\s+the\s+(sky|background|foreground|subject|person|object|colors?|lighting))',
                r'\b(add\s+a\s+(sunset|mountain|person|tree|building|cat|dog|car|rain|snow))',
                r'\b(now\s+make\s+it|then\s+make\s+it|also\s+make\s+it)',
            ],
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)',
                r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))',
                r'\b(code\s+(for|that|this|to|which|example))',
                r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))',
                r'\b(scaffold|boilerplate|template)\s+(for|a)',
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)',
                r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)',
                r'\b(code\s+quality|technical\s+debt|code\s+smell)',
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)',
                r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring))',
                r'\b(error|exception|traceback|stack\s+trace)\s*[:\n]',
                r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)',
            ],
            IntentCategory.CODE_EXECUTION: [
                r'\b(run|execute|eval)\s+(this|the|my)?\s*(code|script|program)',
                r'\b(code\s+execution|execute\s+code)',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
            ],
            IntentCategory.DATA_ANALYSIS: [
                r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)',
                r'\b(statistics?|statistical)\s+(analysis|test|summary|overview)',
                r'\b(insights?\s+(from|in|about|into))',
                r'\b(eda|exploratory\s+data\s+analysis)',
            ],
            IntentCategory.DATA_VISUALIZATION: [
                r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)',
                r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap)',
                r'\b(matplotlib|seaborn|plotly|d3|chart\.js|ggplot|altair)',
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page)',
                r'\b(html|css|javascript|typescript|react|vue|angular|next\.js|svelte|tailwind)\b',
                r'\b(frontend|back[- ]end|full[- ]stack)\s*(development|app)?',
            ],
            IntentCategory.API_DEVELOPMENT: [
                r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)',
                r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service)?',
                r'\b(openapi|swagger)',
            ],
            IntentCategory.DATABASE: [
                r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)',
                r'\b(sql|mysql|postgres|mongodb|redis|sqlite)\s*(query|statement)?',
                r'\b(orm|prisma|sqlalchemy|typeorm|drizzle)',
            ],
            IntentCategory.TRANSLATION: [
                r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)',
                r'\b(how\s+(do\s+you|to)\s+say\s+.+\s+in\s+\w+)',
            ],
            IntentCategory.SUMMARIZATION: [
                r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)',
                r'\b(key\s+(points|takeaways|highlights))',
                r'\b(give\s+me\s+(the\s+)?(gist|bottom\s+line|essence))',
            ],
            IntentCategory.EXPLANATION: [
                r'\b(explain|explanation)\s+(to\s+me\s+)?',
                r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+',
                r'\b(how\s+(does|do|did|can|would|should|to))\s+',
                r'\b(why\s+(is|does|do|are|did|can|would))\s+',
            ],
            IntentCategory.CREATIVE_WRITING: [
                r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|song|haiku)',
                r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller)\s*(writing|story)?',
                r'\b(narrative|plot|character|dialogue)\s*(for|development)?',
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem)?',
                r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability)',
                r'\b(integral|derivative|differentiate|integrat)',
                r'\b(prove|proof)\s+(that|this|the)',
            ],
            IntentCategory.RESEARCH: [
                r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)',
                r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper)?',
                r'\b(what\s+(does\s+)?(research|science)\s+say)',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you)[\s!.?]*$',
                r'^(how\s+are\s+you|what\'s\s+up)[\s!.?]*$',
                r'^(bye|goodbye|see\s+you)[\s!.?]*$',
            ],
        }
        self.compiled_patterns = {intent: [re.compile(p, re.IGNORECASE) for p in patterns]
                                  for intent, patterns in self.patterns.items()}

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.IMAGE_GENERATION: ["image", "picture", "photo", "drawing", "illustration",
                "artwork", "painting", "sketch", "graphic", "visual", "render", "thumbnail", "logo",
                "icon", "banner", "poster", "portrait", "landscape", "digital art", "ai art", "flux"],
            IntentCategory.VIDEO_GENERATION: ["video", "clip", "movie", "film", "animation", "motion",
                "gif", "reel", "veo", "sora", "runway", "animated"],
            IntentCategory.AUDIO_GENERATION: ["audio", "sound", "music", "song", "track", "beat",
                "melody", "tune", "podcast", "instrumental", "minimax"],
            IntentCategory.MUSIC_COVER: ["cover", "remix", "acoustic", "reinterpretation", "remake", "reimagine"],
            IntentCategory.MEDIA_MODIFICATION: ["update", "modify", "change", "alter", "adjust", "edit",
                "tweak", "refine", "enhance", "redo", "regenerate", "upscale", "improve"],
            IntentCategory.CODE_GENERATION: ["code", "script", "function", "class", "module", "program",
                "app", "snippet", "algorithm", "library", "package"],
            IntentCategory.CODE_REVIEW: ["review", "refactor", "optimize", "clean up", "best practice"],
            IntentCategory.CODE_DEBUG: ["bug", "error", "issue", "debug", "fix", "crash", "broken"],
            IntentCategory.DOCUMENT_CREATION: ["document", "pdf", "report", "letter", "email", "essay", "paper"],
            IntentCategory.DATA_ANALYSIS: ["data", "dataset", "csv", "excel", "analytics", "statistics"],
            IntentCategory.DATA_VISUALIZATION: ["chart", "graph", "plot", "visualization", "dashboard", "heatmap"],
            IntentCategory.WEB_DEVELOPMENT: ["website", "webpage", "react", "vue", "html", "css", "tailwind"],
            IntentCategory.API_DEVELOPMENT: ["api", "rest", "graphql", "endpoint", "swagger"],
            IntentCategory.DATABASE: ["database", "schema", "sql", "query", "migration", "orm"],
            IntentCategory.TRANSLATION: ["translate", "translation", "localize"],
            IntentCategory.SUMMARIZATION: ["summarize", "summary", "tldr", "brief", "overview"],
            IntentCategory.EXPLANATION: ["explain", "what is", "how does", "why"],
            IntentCategory.CREATIVE_WRITING: ["story", "poem", "fiction", "narrative", "lyrics"],
            IntentCategory.MATHEMATICAL: ["calculate", "compute", "solve", "math", "equation", "proof"],
            IntentCategory.RESEARCH: ["research", "find", "search", "investigate", "study"],
        }

    def _has_negation(self, text: str, keyword_pos: int) -> bool:
        words_before = text[:keyword_pos].lower().split()[-6:]
        return any(neg in " ".join(words_before) for neg in self.negation_words)

    def _calculate_confidence(self, matched_keywords, matched_patterns, text_length) -> float:
        if not matched_keywords and not matched_patterns: return 0.0
        pattern_conf = min(len(matched_patterns) * 0.35, 0.65)
        keyword_conf = min(len(matched_keywords) * 0.12, 0.25)
        bonus = 0.1 if (matched_keywords and matched_patterns) else 0.0
        length_factor = max(0.5, 1.0 - (text_length / 1500) * 0.4)
        return min((pattern_conf + keyword_conf + bonus) * length_factor, 1.0)

    def _are_related_intents(self, i1, i2) -> bool:
        groups = [
            {IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG, IntentCategory.CODE_EXECUTION},
            {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION},
            {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION, IntentCategory.MUSIC_COVER, IntentCategory.MEDIA_MODIFICATION},
            {IntentCategory.WEB_DEVELOPMENT, IntentCategory.API_DEVELOPMENT, IntentCategory.DATABASE},
            {IntentCategory.EXPLANATION, IntentCategory.SUMMARIZATION},
        ]
        return any(i1 in g and i2 in g for g in groups)

    def detect_intents(self, text: str, threshold: float = 0.25) -> List[IntentResult]:
        text_lower = text.lower()
        results = []
        for intent, compiled_patterns in self.compiled_patterns.items():
            matched_keywords, matched_patterns = [], []
            for pattern in compiled_patterns:
                if pattern.search(text): matched_patterns.append(pattern.pattern)
            if intent in self.synonyms:
                for synonym in self.synonyms[intent]:
                    if synonym in text_lower:
                        pos = text_lower.find(synonym)
                        if not self._has_negation(text, pos): matched_keywords.append(synonym)
            if matched_keywords or matched_patterns:
                confidence = self._calculate_confidence(matched_keywords, matched_patterns, len(text))
                if confidence >= threshold:
                    results.append(IntentResult(intent=intent, confidence=confidence, sub_intents=[],
                                                keywords_matched=matched_keywords, patterns_matched=matched_patterns))
        results.sort(key=lambda x: x.confidence, reverse=True)
        if results:
            primary = results[0]
            for r in results[1:]:
                if self._are_related_intents(primary.intent, r.intent): primary.sub_intents.append(r.intent)
        return results[:1] if results else []

    def get_primary_intent(self, text: str) -> Optional[IntentResult]:
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> ActionType:
        intent = self.get_primary_intent(text)
        if not intent: return ActionType.GENERAL
        action_map = {
            IntentCategory.IMAGE_GENERATION: ActionType.IMAGE,
            IntentCategory.VIDEO_GENERATION: ActionType.VIDEO,
            IntentCategory.AUDIO_GENERATION: ActionType.AUDIO,
            IntentCategory.MUSIC_COVER: ActionType.MUSIC_COVER,
            IntentCategory.MEDIA_MODIFICATION: ActionType.MEDIA_MODIFY,
            IntentCategory.CODE_GENERATION: ActionType.CODE,
            IntentCategory.CODE_REVIEW: ActionType.CODE,
            IntentCategory.CODE_DEBUG: ActionType.CODE,
            IntentCategory.CODE_EXECUTION: ActionType.CODE,
            IntentCategory.DOCUMENT_CREATION: ActionType.DOCUMENT,
            IntentCategory.DATA_ANALYSIS: ActionType.DATA,
            IntentCategory.DATA_VISUALIZATION: ActionType.DATA,
            IntentCategory.WEB_DEVELOPMENT: ActionType.WEB,
            IntentCategory.API_DEVELOPMENT: ActionType.API,
            IntentCategory.DATABASE: ActionType.DATABASE,
            IntentCategory.TRANSLATION: ActionType.TRANSLATION,
            IntentCategory.SUMMARIZATION: ActionType.SUMMARY,
            IntentCategory.EXPLANATION: ActionType.EXPLANATION,
            IntentCategory.CREATIVE_WRITING: ActionType.CREATIVE,
            IntentCategory.MATHEMATICAL: ActionType.MATH,
            IntentCategory.RESEARCH: ActionType.RESEARCH,
            IntentCategory.CONVERSATION: ActionType.CONVERSATION,
        }
        return action_map.get(intent.intent, ActionType.GENERAL)

    def get_required_tools(self, text: str) -> List[str]:
        intent = self.get_primary_intent(text)
        if not intent: return ["llm"]
        tool_map = {
            IntentCategory.IMAGE_GENERATION: ["image_gen", "llm"],
            IntentCategory.VIDEO_GENERATION: ["video_gen", "llm"],
            IntentCategory.AUDIO_GENERATION: ["audio_gen", "llm"],
            IntentCategory.MUSIC_COVER: ["music_cover_gen", "llm"],
            IntentCategory.MEDIA_MODIFICATION: ["media_modify", "llm"],
            IntentCategory.CODE_GENERATION: ["code_exec", "llm"],
            IntentCategory.CODE_REVIEW: ["llm"],
            IntentCategory.CODE_DEBUG: ["code_exec", "llm"],
            IntentCategory.CODE_EXECUTION: ["code_exec"],
            IntentCategory.DOCUMENT_CREATION: ["doc_gen", "llm"],
            IntentCategory.DATA_ANALYSIS: ["code_exec", "data_processing", "llm"],
            IntentCategory.DATA_VISUALIZATION: ["code_exec", "llm"],
            IntentCategory.WEB_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.API_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.DATABASE: ["database", "code_exec", "llm"],
            IntentCategory.TRANSLATION: ["llm"],
            IntentCategory.SUMMARIZATION: ["llm"],
            IntentCategory.EXPLANATION: ["llm"],
            IntentCategory.CREATIVE_WRITING: ["llm"],
            IntentCategory.MATHEMATICAL: ["code_exec", "llm"],
            IntentCategory.RESEARCH: ["web_search", "llm"],
            IntentCategory.CONVERSATION: ["llm"],
        }
        tools = list(tool_map.get(intent.intent, ["llm"]))
        for sub in intent.sub_intents:
            for tool in tool_map.get(sub, []):
                if tool not in tools: tools.append(tool)
        return tools

    def get_code_system_prompt(self, text: str) -> str:
        base = get_system_prompt(text)
        intent = self.get_primary_intent(text)
        if not intent: return base + "\n\nYou are also a helpful coding assistant."
        sub_prompts = {
            IntentCategory.CODE_DEBUG: "\n\nYou are also an expert debugger. Identify root causes, explain WHY, provide exact fixes with code blocks, suggest prevention.",
            IntentCategory.CODE_REVIEW: "\n\nYou are also a senior code reviewer. Review quality, bugs, performance, best practices, security. Be specific and actionable.",
            IntentCategory.CODE_GENERATION: "\n\nYou are also an expert software engineer. Write clean, production-ready code with error handling and comments. Consider edge cases.",
            IntentCategory.WEB_DEVELOPMENT: "\n\nYou are also a full-stack web developer. Use modern best practices, responsive design, accessibility. Provide complete, ready-to-use code.",
            IntentCategory.API_DEVELOPMENT: "\n\nYou are also an API development expert. Follow RESTful principles, proper error handling, input validation, security, clear docs.",
            IntentCategory.DATABASE: "\n\nYou are also a database expert. Design efficient schemas, optimized queries, proper indexes, data integrity. Provide complete SQL/ORM code.",
        }
        return base + sub_prompts.get(intent.intent, "\n\nYou are also a helpful coding assistant.")


_detector = None
def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None: _detector = AdvancedIntentDetector()
    return _detector

def is_image_request(prompt: str) -> bool: return get_detector().get_action_type(prompt) == ActionType.IMAGE
def is_video_request(prompt: str) -> bool: return get_detector().get_action_type(prompt) == ActionType.VIDEO
def is_code_request(prompt: str) -> bool: return get_detector().get_action_type(prompt) == ActionType.CODE
def detect_intent(prompt: str) -> Optional[IntentResult]: return get_detector().get_primary_intent(prompt)
def get_action_type(prompt: str) -> ActionType: return get_detector().get_action_type(prompt)
def get_required_tools(prompt: str) -> List[str]: return get_detector().get_required_tools(prompt)


# =========================
# GEMINI API INTEGRATION (500B+ MODEL)
# =========================
def convert_messages_to_gemini(messages: list, system_prompt: str) -> Tuple[str, list]:
    """Convert OpenAI-style messages to Gemini format. Returns (system_instruction, gemini_contents)"""
    gemini_contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system": continue  # handled separately
        gemini_role = "model" if role == "assistant" else "user"
        if isinstance(content, str):
            gemini_contents.append({"role": gemini_role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # base64 inline image
                            try:
                                header, data = url.split(",", 1)
                                mime = header.split(";")[0].split(":")[1]
                                parts.append({"inlineData": {"mimeType": mime, "data": data}})
                            except: pass
                        else:
                            parts.append({"text": f"[Image URL: {url}]"})
            if parts: gemini_contents.append({"role": gemini_role, "parts": parts})
    return system_prompt, gemini_contents


async def stream_gemini_chat(messages: list, model: str = GEMINI_PRO_MODEL,
                              system_prompt: str = "", max_tokens: int = 8192,
                              temperature: float = 0.7):
    """Stream chat using Google Gemini API (2.5 Pro — 500B+)"""
    if not GOOGLE_API_KEY:
        async for token in stream_groq_chat(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens):
            yield token
        return

    sys_instr, gemini_contents = convert_messages_to_gemini(messages, system_prompt)

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.95,
            "maxOutputTokens": max_tokens,
        }
    }
    if sys_instr:
        payload["systemInstruction"] = {"parts": [{"text": sys_instr}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={GOOGLE_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"Gemini API Error {resp.status_code}: {error_text}")
                    # Fallback to Groq
                    async for token in stream_groq_chat(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens):
                        yield token
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if not data_str.strip(): continue
                        try:
                            chunk = json.loads(data_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text: yield text
                        except json.JSONDecodeError:
                            pass
    except httpx.ConnectError:
        logger.error("Failed to connect to Gemini API, falling back to Groq")
        async for token in stream_groq_chat(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens):
            yield token
    except Exception as e:
        logger.error(f"Gemini stream error: {e}, falling back to Groq")
        async for token in stream_groq_chat(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens):
            yield token


async def gemini_chat_complete(messages: list, model: str = GEMINI_PRO_MODEL,
                                system_prompt: str = "", max_tokens: int = 8192) -> str:
    """Non-streaming Gemini chat completion"""
    if not GOOGLE_API_KEY:
        return await groq_chat_complete(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens)

    sys_instr, gemini_contents = convert_messages_to_gemini(messages, system_prompt)
    payload = {
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.7, "topP": 0.95, "maxOutputTokens": max_tokens}
    }
    if sys_instr:
        payload["systemInstruction"] = {"parts": [{"text": sys_instr}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        if r.status_code != 200:
            logger.error(f"Gemini API Error: {r.status_code} {r.text}")
            return await groq_chat_complete(messages, model=GROQ_FALLBACK_MODEL, max_tokens=max_tokens)
        data = r.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        return ""


async def gemini_vision_analyze(image_base64: str, mime_type: str, prompt: str = "Describe this image in detail. What do you see?") -> str:
    """Analyze an image using Gemini Vision"""
    if not GOOGLE_API_KEY:
        return "Vision analysis unavailable — Google API key not configured."

    payload = {
        "contents": [{
            "role": "user", "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": image_base64}}
            ]
        }],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_PRO_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                logger.error(f"Gemini Vision Error: {r.status_code}")
                return "Failed to analyze image."
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates:
                return "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []))
            return "No analysis available."
    except Exception as e:
        logger.error(f"Gemini Vision error: {e}")
        return f"Vision analysis error: {str(e)}"


# =========================
# GROQ API (FALLBACK)
# =========================
def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def get_elevenlabs_headers():
    return {"xi-api-key": ELEVENLABS_API_KEY}


async def stream_groq_chat(messages: list, model: str = GROQ_FALLBACK_MODEL, max_tokens: int = 1024):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"Groq API Error {resp.status_code}: {error_text}")
                    raise Exception(f"AI Service Error ({resp.status_code})")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]": break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta: yield delta
                        except: pass
    except httpx.ConnectError:
        raise Exception("Connection to AI service failed.")
    except Exception as e:
        if "AI Service Error" in str(e): raise e
        logger.error(f"Groq stream error: {e}")
        raise e


async def groq_chat_complete(messages: list, model: str = GROQ_FALLBACK_MODEL, max_tokens: int = 1024) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": model, "messages": messages, "max_tokens": max_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# =========================
# REPLICATE API — MEDIA GENERATION
# =========================
async def create_replicate_prediction(model: str, input_data: dict) -> Dict:
    """Create a Replicate prediction and return result (polling until complete)"""
    if not REPLICATE_API_TOKEN:
        raise HTTPException(500, "Replicate API Token not configured. Media generation unavailable.")

    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json",
               "Prefer": "wait"}

    url = f"https://api.replicate.com/v1/models/{model}/predictions"

    async with httpx.AsyncClient(timeout=300) as client:
        # Create prediction
        r = await client.post(url, headers=headers, json={"input": input_data})

        if r.status_code == 201:
            data = r.json()
            prediction_id = data.get("id")
            status = data.get("status", "starting")

            # If already completed (some models are fast)
            if status == "succeeded":
                return {"status": "succeeded", "output": data.get("output"), "id": prediction_id}

            # Poll for completion
            max_polls = 120  # up to ~2 minutes
            poll_interval = 2

            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
                poll_r = await client.get(poll_url, headers=headers)

                if poll_r.status_code == 200:
                    poll_data = poll_r.json()
                    poll_status = poll_data.get("status")

                    if poll_status == "succeeded":
                        return {"status": "succeeded", "output": poll_data.get("output"), "id": prediction_id}
                    elif poll_status == "failed":
                        error = poll_data.get("error", "Unknown error")
                        return {"status": "failed", "error": error, "id": prediction_id}
                    elif poll_status in ("canceled", "cancelled"):
                        return {"status": "cancelled", "id": prediction_id}
                    # still processing — continue polling
                else:
                    logger.warning(f"Replicate poll error: {poll_r.status_code}")
                    return {"status": "timeout", "id": prediction_id, "error": "Generation timed out"}

        elif r.status_code == 200:
            # Some responses come as 200 directly
            data = r.json()
            return {"status": "succeeded", "output": data.get("output"), "id": data.get("id")}

        else:
            logger.error(f"Replicate create error: {r.status_code} {r.text}")
            raise HTTPException(500, f"Media generation failed: {r.status_code}")


async def generate_image_flux(prompt: str, aspect_ratio: str = "1:1",
                               num_outputs: int = 1, output_format: str = "webp") -> Dict:
    """Generate image using black-forest-labs/flux-2-max on Replicate"""
    input_data = {
        "prompt": prompt,
        "num_outputs": num_outputs,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "output_quality": 95,
        "prompt_strength": 0.8,
    }
    return await create_replicate_prediction(REPLICATE_IMAGE_MODEL, input_data)


async def generate_video_veo(prompt: str, aspect_ratio: str = "16:9",
                              duration: str = "5") -> Dict:
    """Generate video using google/veo-3.1-lite on Replicate"""
    input_data = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }
    return await create_replicate_prediction(REPLICATE_VIDEO_MODEL, input_data)


async def generate_music_minimax(prompt: str, duration: int = 30) -> Dict:
    """Generate music using minimax/music-2.6 on Replicate"""
    input_data = {
        "prompt": prompt,
        "duration": duration,
    }
    return await create_replicate_prediction(REPLICATE_MUSIC_MODEL, input_data)


async def generate_music_cover_minimax(prompt: str, reference_audio_url: str = None) -> Dict:
    """Generate music cover using minimax/music-cover on Replicate"""
    input_data = {"prompt": prompt}
    if reference_audio_url:
        input_data["reference_audio"] = reference_audio_url
    return await create_replicate_prediction(REPLICATE_MUSIC_COVER_MODEL, input_data)


# =========================
# MEDIA CONTEXT SYSTEM (ALL-KNOWING)
# =========================
def store_media_context(conv_id: str, media_type: str, url: str, prompt: str, description: str):
    """Store a media generation in context store so AI remembers it"""
    if conv_id not in _media_context_store:
        _media_context_store[conv_id] = []
    entry = {
        "type": media_type,  # "image", "video", "audio", "music_cover"
        "url": url,
        "prompt": prompt,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    _media_context_store[conv_id].append(entry)
    # Keep last 20 media items per conversation
    if len(_media_context_store[conv_id]) > 20:
        _media_context_store[conv_id] = _media_context_store[conv_id][-20:]


def get_media_context(conv_id: str) -> List[Dict[str, Any]]:
    """Get all media context for a conversation"""
    return _media_context_store.get(conv_id, [])


def get_latest_media(conv_id: str, media_type: str = None) -> Optional[Dict[str, Any]]:
    """Get most recent media item, optionally filtered by type"""
    items = _media_context_store.get(conv_id, [])
    if not items: return None
    if media_type:
        filtered = [m for m in items if m["type"] == media_type]
        return filtered[-1] if filtered else None
    return items[-1]


def build_media_context_prompt(conv_id: str) -> str:
    """Build a natural language description of all media generated in this conversation"""
    items = get_media_context(conv_id)
    if not items: return ""

    parts = ["\n\nMEDIA CONTEXT — You have previously generated following media in this conversation:"]
    for i, item in enumerate(items, 1):
        parts.append(
            f"{i}. [{item['type'].upper()}] Prompt: \"{item['prompt']}\" | "
            f"Description: \"{item['description']}\" | URL: {item['url']}"
        )
    parts.append(
        "When user asks to modify, update, or reference any of these, use context above. "
        "You know exactly what each piece of media contains because you created prompts for them."
    )
    return "\n".join(parts)


# =========================
# EXTERNAL TOOLS
# =========================
async def solve_math(query: str) -> str:
    if wolframalpha and WOLFRAM_ALPHA_API_KEY:
        try:
            client = wolframalpha.Client(WOLFRAM_ALPHA_API_KEY)
            res = client.query(query)
            if hasattr(res, 'results') and res.results:
                return next(res.results).text
            elif hasattr(res, 'pods') and len(res.pods) > 1:
                return res.pods[1].text if res.pods[1].text else "Wolfram could not solve this."
            return "Wolfram could not solve this."
        except Exception as e:
            logger.warning(f"Wolfram failed: {e}, falling back to LLM")
    # Fallback to LLM
    messages = [{"role": "user", "content": f"Solve step by step: {query}"}]
    return await gemini_chat_complete(messages, system_prompt="You are a math expert. Solve step by step with clear notation.")


async def search_google(query: str) -> str:
    if not SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY not configured.")
        return ""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": 5}
            response = await client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("organic_results", [])
            formatted = []
            for res in results[:5]:
                formatted.append(f"Title: {res.get('title', 'N/A')}\nLink: {res.get('link', '#')}\nSnippet: {res.get('snippet', '')}")
            return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"SerpApi search failed: {e}")
        return ""


JUDGE0_LANGUAGES = {
    "python": 71, "javascript": 63, "java": 62, "cpp": 54,
    "c": 50, "csharp": 51, "go": 60, "rust": 73, "sql": 82
}

async def execute_code(code: str, language: str) -> dict:
    if not JUDGE0_API_KEY: return {"error": "Judge0 API Key not configured"}
    lang_id = JUDGE0_LANGUAGES.get(language.lower(), 71)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://judge0-ce.p.rapidapi.com/submissions",
            headers={"X-RapidAPI-Key": JUDGE0_API_KEY, "Content-Type": "application/json"},
            json={"source_code": code, "language_id": lang_id, "stdin": ""})
        if r.status_code != 201: return {"error": f"Judge0 Submit Failed: {r.status_code}"}
        token = r.json()["token"]
        for _ in range(10):
            await asyncio.sleep(1)
            r = await client.get(f"https://judge0-ce.p.rapidapi.com/submissions/{token}",
                headers={"X-RapidAPI-Key": JUDGE0_API_KEY})
            data = r.json()
            if data["status"]["id"] in (1, 2): continue
            return {"stdout": data.get("stdout", ""), "stderr": data.get("stderr", ""), "status": data["status"]["description"]}
    return {"error": "Execution timed out"}


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True
    model: Optional[str] = None


class RegenerateRequest(BaseModel):
    conversation_id: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "rachel"


class MediaGenerateRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    aspect_ratio: str = "1:1"
    duration: Optional[int] = 30
    reference_audio_url: Optional[str] = None


# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _execute_supabase_with_retry(query_builder, description="Supabase Operation"):
    max_retries = 3
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(query_builder.execute)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            if "502" in error_str or "Bad Gateway" in error_str or "Expecting value" in error_str:
                logger.warning(f"{description} retry {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            else:
                logger.error(f"{description} failed: {e}")
                break
    if last_exception: raise last_exception


async def get_user(request: Request, response: Response, remember: Optional[bool] = None) -> Dict[str, Any]:
    await cleanup_session_cache()
    primary_id = request.cookies.get(PRIMARY_COOKIE)
    backup_id = request.cookies.get(BACKUP_COOKIE)
    device_cookie = request.cookies.get(DEVICE_COOKIE)
    stored_fingerprint = request.cookies.get(FINGERPRINT_COOKIE)
    session_token = request.cookies.get(SESSION_TOKEN_COOKIE)
    session_expiry = request.cookies.get(SESSION_EXPIRY_COOKIE)
    current_fingerprint = generate_device_fingerprint(request)
    if remember is None: remember = not is_session_expired(session_expiry or "0")

    user_obj = {"id": None, "email": None, "memory": "", "fingerprint": current_fingerprint,
                "session_valid": False, "session_token": None}
    user_id = None

    if primary_id and session_token:
        if is_session_expired(session_expiry or "0"):
            clear_session_cookies(response)
        else:
            token_valid = await validate_session_token(primary_id, session_token)
            if token_valid:
                user_id = primary_id
                user_obj["session_valid"] = True
                user_obj["session_token"] = session_token
                if should_refresh_session(session_expiry or "0"):
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
            else:
                logger.warning(f"Invalid session token for user {primary_id[:8]}...")

    if not user_id and backup_id:
        user_id = backup_id

    if not user_id and device_cookie:
        try:
            fp_part = device_cookie.split("_")[0] if "_" in device_cookie else device_cookie
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1),
                description="User Lookup by Fingerprint")
            if fp_resp.data: user_id = fp_resp.data[0]["id"]
        except: pass

    if not user_id and stored_fingerprint:
        try:
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", stored_fingerprint).limit(1),
                description="User Lookup by Stored FP")
            if fp_resp.data: user_id = fp_resp.data[0]["id"]
        except: pass

    if user_id:
        try:
            user_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", user_id).limit(1),
                description="User Lookup by ID"
            )
            if user_resp.data:
                u = user_resp.data[0]
                user_obj = {"id": u["id"], "email": u.get("email"), "memory": u.get("memory", ""),
                            "is_premium": u.get("is_premium", False), "is_lifetime": u.get("is_lifetime", False),
                            "plan": u.get("plan", "free"), "fingerprint": current_fingerprint,
                            "session_valid": user_obj.get("session_valid", False),
                            "session_token": user_obj.get("session_token")}
                if u.get("fingerprint") != current_fingerprint:
                    try:
                        await _execute_supabase_with_retry(
                            supabase.table("users").update({"fingerprint": current_fingerprint}).eq("id", user_id),
                            description="Update Fingerprint"
                        )
                    except: pass
                if not user_obj["session_valid"]:
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
                    user_obj["session_valid"] = True
                set_session_cookies(response, user_id, current_fingerprint, user_obj["session_token"], remember)
                return user_obj
        except Exception as e:
            logger.error(f"User data fetch failed: {e}")

    new_id = str(uuid.uuid4())
    try:
        await _execute_supabase_with_retry(
            supabase.table("users").upsert(
                {"id": new_id, "email": f"anon+{new_id[:8]}@local", "memory": "", "fingerprint": current_fingerprint},
                on_conflict="id"),
            description="Create Anonymous User")
        )
        user_obj["id"] = new_id
    except Exception as e:
        logger.error(f"Failed to create anonymous user: {e}")
        user_obj["id"] = new_id

    new_token = await create_user_session(new_id, current_fingerprint, remember)
    user_obj["session_token"] = new_token
    user_obj["session_valid"] = True
    set_session_cookies(response, new_id, current_fingerprint, new_token, remember)
    return user_obj


async def update_user_memory(user_id: str, new_memory: str):
    try:
        await _execute_supabase_with_retry(
            supabase.table("users").update({"memory": new_memory}).eq("id", user_id),
            description="Update User Memory"
        )
        if user_id in _session_cache: _session_cache[user_id]["memory"] = new_memory
    except Exception as e:
        logger.error(f"Failed to update user memory: {e}")


async def get_history(conv_id: str) -> list:
    try:
        result = await _execute_supabase_with_retry(
            supabase.table("messages").select("role, content, content_type")
            .eq("conversation_id", conv_id).order("created_at", desc=False).limit(30),
            description="Fetch History"
        )
        if result.data: 
            return [{"role": m["role"], "content": m["content"], "content_type": m.get("content_type", "text")} for m in result.data]
        return []
    except: 
        return []


async def save_message(user_id: str, conv_id: str, role: str, content: str, metadata: dict = None):
    try:
        payload = {
            "id": str(uuid.uuid4()), 
            "conversation_id": conv_id, 
            "role": role, 
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        if metadata: payload["metadata"] = json.dumps(metadata)
        
        await _execute_supabase_with_retry(
            supabase.table("messages").insert(payload),
            description="Save Message"
        )
    except Exception as e:
        logger.error(f"Failed to save message: {e}")


# =========================
# UNIFIED LLM STREAM (auto-selects Gemini or Groq)
# =========================
async def unified_stream_chat(messages: list, model: str = PRIMARY_LLM_MODEL,
                               system_prompt: str = "", max_tokens: int = 8192):
    """Stream chat, automatically using Gemini or Groq based on model name"""
    if is_gemini_model(model) and GOOGLE_API_KEY:
        async for token in stream_gemini_chat(messages, model=model, system_prompt=system_prompt, max_tokens=max_tokens):
            yield token
    else:
        async for token in stream_groq_chat(messages, model=model, max_tokens=max_tokens):
            yield token


async def unified_chat_complete(messages: list, model: str = PRIMARY_LLM_MODEL,
                                 system_prompt: str = "", max_tokens: int = 8192) -> str:
    """Non-streaming chat, auto-selects backend"""
    if is_gemini_model(model) and GOOGLE_API_KEY:
        return await gemini_chat_complete(messages, model=model, system_prompt=system_prompt, max_tokens=max_tokens)
    else:
        return await groq_chat_complete(messages, model=model, max_tokens=max_tokens)


# =========================
# MEDIA GENERATION PROMPT BUILDER
# =========================
async def build_media_prompt(user_prompt: str, media_type: str, conv_id: str = None,
                              modifier_context: str = "") -> str:
    """Use LLM to craft an optimized prompt for media generation"""
    media_context = build_media_context_prompt(conv_id) if conv_id else ""

    system = f"""You are an expert {media_type} prompt engineer. Your job is to take user's request and create an optimized, detailed prompt for an AI {media_type} generator.

Rules:
- Be specific and descriptive
- Include style, mood, lighting, composition details
- For images: describe visual elements, art style, colors, camera angle
- For videos: describe motion, scene progression, timing, visual style
- For music: describe genre, instruments, tempo, mood, key, arrangement
- Keep prompt focused and under 200 words
- Output ONLY the prompt text, nothing else
{media_context}
{modifier_context}"""

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    result = await unified_chat_complete(messages, model=FAST_LLM_MODEL, max_tokens=300, system_prompt=system)
    return result.strip().strip('"').strip("'")


# =========================
# MAIN ENDPOINT: /ask/universal
# =========================
@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")
    body = {}
    remember = True
    uploaded_image_b64 = None
    uploaded_image_mime = None

    if "application/json" in content_type:
        try:
            body = await req.json()
            remember = body.get("remember", True)
        except: raise HTTPException(400, "Invalid JSON")

    elif "multipart/form-data" in content_type:
        try:
            form = await req.form()
            body = {}
            for key, value in form.items():
                if key == "file" and isinstance(value, UploadFile):
                    file_content = await value.read()
                    filename = value.filename or "upload"
                    file_cat = get_file_category(filename)

                    if file_cat == FileCategory.IMAGE:
                        # Store for vision analysis
                        mime = value.content_type or mimetypes.guess_type(filename)[0] or "image/png"
                        uploaded_image_b64 = base64.b64encode(file_content).decode("utf-8")
                        uploaded_image_mime = mime
                        body["prompt"] = form.get("prompt", "Analyze this image")
                    else:
                        result = await extract_file_content(file_content, filename)
                        body["prompt"] = form.get("prompt", "") + "\n\n" + result.content
                else:
                    body[key] = value
            remember = body.get("remember", True)
        except: raise HTTPException(400, "Invalid Form Data")
    else:
        raise HTTPException(415, f"Unsupported content-type: {content_type}")

    user = await get_user(req, res, remember=remember)
    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    request_model = body.get("model")
    api_model = resolve_model(request_model)

    if not prompt: raise HTTPException(400, "Prompt required")

    # Conversation setup
    conversation_exists = False
    if conv_id:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", conv_id).limit(1),
            description="Check Conversation"
        )
        if check.data: conversation_exists = True
        else: conv_id = str(uuid.uuid4())
    if not conv_id: conv_id = str(uuid.uuid4())
    if not conversation_exists:
        try:
            await _execute_supabase_with_retry(
                supabase.table("conversations").insert({
                    "id": conv_id, "user_id": user["id"],
                    "title": prompt[:30], "created_at": datetime.now(timezone.utc).isoformat()
                }), description="Create Conversation")
        except: pass

    try: await save_message(user["id"], conv_id, "user", prompt)
    except: pass

    # Detect intent
    intent_result = detect_intent(prompt)
    action_type = get_action_type(prompt)
    required_tools = get_required_tools(prompt)

    if intent_result:
        logger.info(f"[INTENT] action={action_type.value} intent={intent_result.intent.value} conf={intent_result.confidence:.2%} tools={required_tools}")
    else:
        logger.info(f"[INTENT] action={action_type.value} (no specific intent)")

    # ===== IMAGE ANALYSIS (uploaded image) =====
    if uploaded_image_b64:
        async def vision_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Analyzing image with Gemini Vision..."})
                analysis = await gemini_vision_analyze(uploaded_image_b64, uploaded_image_mime, prompt)
                # Store in media context
                store_media_context(conv_id, "image_analyzed", "uploaded",
                                    prompt, analysis[:500])
                for char in analysis:
                    if task.cancelled(): break
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.005)
                await save_message(user["id"], conv_id, "assistant", analysis)
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Vision error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(vision_gen(), media_type="text/event-stream")

    # ===== IMAGE GENERATION (Flux 2 Max) =====
    if action_type == ActionType.IMAGE:
        async def image_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Crafting your image prompt..."})
                image_prompt = await build_media_prompt(prompt, "image", conv_id)
                yield sse({"type": "status", "message": f"Generating image with Flux 2 Max: \"{image_prompt[:80]}...\""})
                result = await generate_image_flux(image_prompt)
                if result["status"] == "succeeded":
                    output = result["output"]
                    # Replicate Flux returns a list of URLs
                    image_url = output[0] if isinstance(output, list) else output
                    description = f"AI-generated image: {image_prompt}"
                    store_media_context(conv_id, "image", image_url, image_prompt, description)

                    # Emit Media Event (Frontend expects 'type': 'media')
                    yield sse({
                        "type": "media", 
                        "media_type": "image", 
                        "url": image_url,
                        "prompt": image_prompt,
                        "description": description
                    })

                    # Then stream text explanation
                    desc_msg = f"I've generated an image for you! The prompt was: \"{image_prompt}\"\n\n"
                    for char in desc_msg:
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)
                    
                    await save_message(user["id"], conv_id, "assistant", desc_msg + image_url)
                else:
                    error = result.get("error", "Unknown error")
                    yield sse({"type": "error", "message": f"Image generation failed: {error}"})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Image gen error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(image_gen(), media_type="text/event-stream")

    # ===== VIDEO GENERATION (Veo 3.1 Lite) =====
    if action_type == ActionType.VIDEO:
        async def video_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Crafting your video prompt..."})
                video_prompt = await build_media_prompt(prompt, "video", conv_id)
                yield sse({"type": "status", "message": f"Generating video with Google Veo 3.1 Lite: \"{video_prompt[:80]}...\""})
                result = await generate_video_veo(video_prompt)
                if result["status"] == "succeeded":
                    output = result["output"]
                    video_url = output[0] if isinstance(output, list) else output
                    description = f"AI-generated video: {video_prompt}"
                    store_media_context(conv_id, "video", video_url, video_prompt, description)

                    # Emit Media Event
                    yield sse({
                        "type": "media", 
                        "media_type": "video", 
                        "url": video_url,
                        "prompt": video_prompt,
                        "description": description
                    })

                    desc_msg = f"I've generated a video for you! The prompt was: \"{video_prompt}\"\n\n"
                    for char in desc_msg:
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)

                    await save_message(user["id"], conv_id, "assistant", desc_msg + video_url)
                else:
                    yield sse({"type": "error", "message": f"Video generation failed: {result.get('error', 'Unknown')}"})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Video gen error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(video_gen(), media_type="text/event-stream")

    # ===== AUDIO / MUSIC GENERATION (MiniMax Music 2.6) =====
    if action_type == ActionType.AUDIO:
        async def audio_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Crafting your music prompt..."})
                music_prompt = await build_media_prompt(prompt, "music", conv_id)
                yield sse({"type": "status", "message": f"Generating music with MiniMax Music 2.6: \"{music_prompt[:80]}...\""})
                result = await generate_music_minimax(music_prompt)
                if result["status"] == "succeeded":
                    output = result["output"]
                    audio_url = output[0] if isinstance(output, list) else output
                    description = f"AI-generated music: {music_prompt}"
                    store_media_context(conv_id, "audio", audio_url, music_prompt, description)

                    # Emit Media Event
                    yield sse({
                        "type": "media", 
                        "media_type": "audio", 
                        "url": audio_url,
                        "prompt": music_prompt,
                        "description": description
                    })

                    desc_msg = f"I've generated music for you! The prompt was: \"{music_prompt}\"\n\n"
                    for char in desc_msg:
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)

                    await save_message(user["id"], conv_id, "assistant", desc_msg + audio_url)
                else:
                    yield sse({"type": "error", "message": f"Music generation failed: {result.get('error', 'Unknown')}"})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Music gen error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(audio_gen(), media_type="text/event-stream")

    # ===== MUSIC COVER (MiniMax Music Cover) =====
    if action_type == ActionType.MUSIC_COVER:
        async def cover_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Crafting your music cover prompt..."})
                cover_prompt = await build_media_prompt(prompt, "music cover", conv_id)
                yield sse({"type": "status", "message": f"Generating music cover with MiniMax Music Cover: \"{cover_prompt[:80]}...\""})
                result = await generate_music_cover_minimax(cover_prompt)
                if result["status"] == "succeeded":
                    output = result["output"]
                    audio_url = output[0] if isinstance(output, list) else output
                    description = f"AI-generated music cover: {cover_prompt}"
                    store_media_context(conv_id, "music_cover", audio_url, cover_prompt, description)

                    # Emit Media Event
                    yield sse({
                        "type": "media", 
                        "media_type": "audio", 
                        "url": audio_url,
                        "prompt": cover_prompt,
                        "description": description
                    })

                    desc_msg = f"I've generated a music cover for you! Prompt: \"{cover_prompt}\"\n\n"
                    for char in desc_msg:
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)

                    await save_message(user["id"], conv_id, "assistant", desc_msg + audio_url)
                else:
                    yield sse({"type": "error", "message": f"Music cover generation failed: {result.get('error', 'Unknown')}"})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Music cover gen error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(cover_gen(), media_type="text/event-stream")

    # ===== MEDIA MODIFICATION =====
    if action_type == ActionType.MEDIA_MODIFY:
        async def modify_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                # Find most recent media to modify
                latest = get_latest_media(conv_id)
                if not latest:
                    # If no media to modify, treat as new gen request based on keywords
                    # Check intent again just in case
                    sub_intent = detect_intent(prompt)
                    if sub_intent and sub_intent.intent == IntentCategory.VIDEO_GENERATION:
                        modified_prompt = await build_media_prompt(prompt, "video", conv_id)
                        result = await generate_video_veo(modified_prompt)
                    elif sub_intent and sub_intent.intent == IntentCategory.AUDIO_GENERATION:
                        modified_prompt = await build_media_prompt(prompt, "music", conv_id)
                        result = await generate_music_minimax(modified_prompt)
                    else:
                        # Default to image
                        modified_prompt = await build_media_prompt(prompt, "image", conv_id)
                        result = await generate_image_flux(modified_prompt)
                else:
                    media_type = latest["type"]
                    original_prompt = latest["prompt"]
                    original_description = latest["description"]

                    modifier_context = f"""
The user wants to modify a previously generated {media_type}.
Original prompt: "{original_prompt}"
Original description: "{original_description}"
User's modification request: "{prompt}"

Create a NEW prompt that incorporates user's modifications into original concept."""

                    yield sse({"type": "status", "message": f"Updating your {media_type}..."})

                    if media_type in ("image", "image_analyzed"):
                        modified_prompt = await build_media_prompt(prompt, "image", conv_id, modifier_context)
                        result = await generate_image_flux(modified_prompt)
                    elif media_type == "video":
                        modified_prompt = await build_media_prompt(prompt, "video", conv_id, modifier_context)
                        result = await generate_video_veo(modified_prompt)
                    elif media_type in ("audio", "music_cover"):
                        modified_prompt = await build_media_prompt(prompt, "music", conv_id, modifier_context)
                        result = await generate_music_minimax(modified_prompt)
                    else:
                        modified_prompt = await build_media_prompt(prompt, "image", conv_id, modifier_context)
                        result = await generate_image_flux(modified_prompt)

                if result["status"] == "succeeded":
                    output = result["output"]
                    media_url = output[0] if isinstance(output, list) else output
                    actual_type = media_type if media_type != "image_analyzed" else "image"
                    description = f"Updated {actual_type}: {modified_prompt}"
                    store_media_context(conv_id, actual_type, media_url, modified_prompt, description)

                    # Emit Media Update Event
                    yield sse({
                        "type": "media", 
                        "media_type": actual_type, 
                        "url": media_url,
                        "prompt": modified_prompt,
                        "description": description,
                        "is_update": True
                    })

                    desc_msg = f"I've updated your {actual_type}! New prompt: \"{modified_prompt}\"\n\n"
                    for char in desc_msg:
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)

                    await save_message(user["id"], conv_id, "assistant", desc_msg + media_url)
                else:
                    yield sse({"type": "error", "message": f"Media update failed: {result.get('error', 'Unknown')}"})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Media modify error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(modify_gen(), media_type="text/event-stream")

    # ===== MATH (Wolfram Alpha) =====
    if intent_result and intent_result.intent == IntentCategory.MATHEMATICAL:
        async def math_gen():
            yield sse({"type": "status", "message": "Calculating with Wolfram Alpha..."})
            try:
                result = await solve_math(prompt)
                for char in result:
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.01)
                await save_message(user["id"], conv_id, "assistant", result)
                yield sse({"type": "done"})
            except Exception as e:
                yield sse({"type": "error", "message": str(e)})
        return StreamingResponse(math_gen(), media_type="text/event-stream")

    # ===== CODE EXECUTION (Judge0) =====
    if intent_result and intent_result.intent == IntentCategory.CODE_EXECUTION:
        match = re.search(r"```(\w+)?\n(.*?)```", prompt, re.DOTALL)
        code = match.group(2) if match else prompt
        lang = match.group(1) if match and match.group(1) else "python"

        async def exec_gen():
            yield sse({"type": "status", "message": f"Executing {lang} code..."})
            try:
                res = await execute_code(code, lang)
                yield sse({"type": "exec_result", "data": res})
                yield sse({"type": "done"})
            except Exception as e:
                yield sse({"type": "error", "message": str(e)})
        return StreamingResponse(exec_gen(), media_type="text/event-stream")

    # ===== RESEARCH (SerpApi) =====
    if intent_result and intent_result.intent == IntentCategory.RESEARCH:
        async def research_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Searching web..."})
                search_results = await search_google(prompt)
                media_ctx = build_media_context_prompt(conv_id)
                if not search_results:
                    # Fallback if search fails or no key
                    system_prompt = get_system_prompt(prompt) + "\n\nYou are a helpful research assistant." + media_ctx
                    user_memory = user.get("memory", "")
                    if user_memory: system_prompt += f"\n\nUser Context: {user_memory}"
                    history = await get_history(conv_id)
                    messages = [{"role": "system", "content": system_prompt}] + history
                    full_text = ""
                    async for token in unified_stream_chat(messages, model=api_model, system_prompt=system_prompt):
                        if task.cancelled(): break
                        full_text += token
                        yield sse({"type": "token", "text": token})
                    
                    new_memory = (user_memory + "\n" + full_text[-500:])[-5000:]
                    asyncio.create_task(update_user_memory(user["id"], new_memory))
                    await save_message(user["id"], conv_id, "assistant", full_text)
                else:
                    system_prompt = get_system_prompt(prompt) + f"""

You are a research assistant. I performed a web search and gathered these results:

{search_results}

Answer user's question based on these results. Cite sources if possible.""" + media_ctx
                    user_memory = user.get("memory", "")
                    if user_memory: system_prompt += f"\n\nUser Context: {user_memory}"
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                    full_text = ""
                    async for token in unified_stream_chat(messages, model=api_model, system_prompt=system_prompt):
                        if task.cancelled(): break
                        full_text += token
                        yield sse({"type": "token", "text": token})

                new_memory = (user_memory + "\n" + full_text[-500:])[-5000:]
                asyncio.create_task(update_user_memory(user["id"], new_memory))
                await save_message(user["id"], conv_id, "assistant", full_text)
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Research stream error: {e}")
                yield sse({"type": "error", "message": "Research failed."})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(research_gen(), media_type="text/event-stream")

    # ===== DEFAULT CHAT (Gemini 2.5 Pro 500B+ / Groq fallback) =====
    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                history = await get_history(conv_id) if conv_id else []
                base_system = get_system_prompt(prompt)
                user_memory = user.get("memory", "")
                if user_memory: base_system += f"\n\nUser Context: {user_memory}"

                # Add media context so AI knows about generated media
                media_ctx = build_media_context_prompt(conv_id)
                if media_ctx: base_system += media_ctx

                full_history = [{"role": "system", "content": base_system}] + history
                full_text = ""

                if is_gemini_model(api_model) and GOOGLE_API_KEY:
                    async for token in stream_gemini_chat(full_history, model=api_model,
                                                          system_prompt=base_system):
                        if task.cancelled(): break
                        full_text += token
                        yield sse({"type": "token", "text": token})
                else:
                    async for token in stream_groq_chat(full_history, model=api_model):
                        if task.cancelled(): break
                        full_text += token
                        yield sse({"type": "token", "text": token})

                new_memory = (user_memory + "\n" + full_text[-500:])[-5000:] if user_memory else full_text[-1000:]
                asyncio.create_task(update_user_memory(user["id"], new_memory))
                try: await save_message(user["id"], conv_id, "assistant", full_text)
                except: pass
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Universal stream error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        history = await get_history(conv_id)
        base_system = get_system_prompt(prompt)
        user_memory = user.get("memory", "")
        if user_memory: base_system += f"\n\nUser Context: {user_memory}"
        media_ctx = build_media_context_prompt(conv_id)
        if media_ctx: base_system += media_ctx
        
        full_history = [{"role": "system", "content": base_system}] + history
        reply = await unified_chat_complete(full_history, model=api_model, system_prompt=base_system)
        new_memory = (user_memory + "\n" + reply[-500:])[-5000:] if user_memory else reply[-1000:]
        asyncio.create_task(update_user_memory(user["id"], new_memory))
        await save_message(user["id"], conv_id, "assistant", reply)
        return {"reply": reply}


# =========================
# DEDICATED MEDIA GENERATION ENDPOINTS
# =========================
@app.post("/generate/image")
async def generate_image_endpoint(req: Request, res: Response):
    """Dedicated image generation endpoint using Flux 2 Max"""
    body = await req.json()
    prompt = body.get("prompt", "")
    aspect_ratio = body.get("aspect_ratio", "1:1")
    conv_id = body.get("conversation_id")
    user = await get_user(req, res)

    if not prompt: raise HTTPException(400, "Prompt required")

    image_prompt = await build_media_prompt(prompt, "image", conv_id)
    result = await generate_image_flux(image_prompt, aspect_ratio=aspect_ratio)

    if result["status"] == "succeeded":
        output = result["output"]
        image_url = output[0] if isinstance(output, list) else output
        description = f"AI-generated image: {image_prompt}"
        if conv_id: store_media_context(conv_id, "image", image_url, image_prompt, description)
        return {"status": "succeeded", "url": image_url, "prompt": image_prompt, "description": description}
    else:
        raise HTTPException(500, f"Image generation failed: {result.get('error', 'Unknown')}")


@app.post("/generate/video")
async def generate_video_endpoint(req: Request, res: Response):
    """Dedicated video generation endpoint using Google Veo 3.1 Lite"""
    body = await req.json()
    prompt = body.get("prompt", "")
    aspect_ratio = body.get("aspect_ratio", "16:9")
    conv_id = body.get("conversation_id")
    user = await get_user(req, res)

    if not prompt: raise HTTPException(400, "Prompt required")

    video_prompt = await build_media_prompt(prompt, "video", conv_id)
    result = await generate_video_veo(video_prompt, aspect_ratio=aspect_ratio)

    if result["status"] == "succeeded":
        output = result["output"]
        video_url = output[0] if isinstance(output, list) else output
        description = f"AI-generated video: {video_prompt}"
        if conv_id: store_media_context(conv_id, "video", video_url, video_prompt, description)
        return {"status": "succeeded", "url": video_url, "prompt": video_prompt, "description": description}
    else:
        raise HTTPException(500, f"Video generation failed: {result.get('error', 'Unknown')}")


@app.post("/generate/music")
async def generate_music_endpoint(req: Request, res: Response):
    """Dedicated music generation endpoint using MiniMax Music 2.6"""
    body = await req.json()
    prompt = body.get("prompt", "")
    duration = body.get("duration", 30)
    conv_id = body.get("conversation_id")
    user = await get_user(req, res)

    if not prompt: raise HTTPException(400, "Prompt required")

    music_prompt = await build_media_prompt(prompt, "music", conv_id)
    result = await generate_music_minimax(music_prompt, duration=duration)

    if result["status"] == "succeeded":
        output = result["output"]
        audio_url = output[0] if isinstance(output, list) else output
        description = f"AI-generated music: {music_prompt}"
        if conv_id: store_media_context(conv_id, "audio", audio_url, music_prompt, description)
        return {"status": "supports", "url": audio_url, "prompt": music_prompt, "description": description}
    else:
        raise HTTPException(500, f"Music generation failed: {result.get('error', 'Unknown')}")


# =========================
# UPDATED FILE ANALYSIS ENDPOINT
# =========================
@app.post("/analyze/file")
async def analyze_file_endpoint(
    req: Request,
    files: List[UploadFile] = File(...),
    prompt: str = Form(""),
    stream: bool = True
):
    """
    Enhanced file analysis endpoint supporting:
    - Up to 5 files (images/videos/documents).
    - Text prompt for analysis (e.g. "How much money is in this picture?").
    - Video duration check (max 60s).
    - Multi-image support.
    """
    user = await get_user(req, Response())
    
    if len(files) > 5:
        raise HTTPException(400, "Maximum of 5 files allowed at a time.")
    
    # Stores base64 data URLs for Gemini
    visual_parts = [] 
    text_parts = []
    video_count = 0
    
    for uploaded_file in files:
        content = await uploaded_file.read()
        filename = uploaded_file.filename or "unknown"
        content_type = uploaded_file.content_type or ""
        file_size = len(content)
        
        logger.info(f"[FILE] Upload: {filename} ({format_file_size(file_size)}, type={content_type})")
        
        if not content:
            continue

        # --- VIDEO HANDLING ---
        if content_type.startswith("video/") or get_file_category(filename) == FileCategory.VIDEO:
            video_count += 1
            if video_count > 1:
                raise HTTPException(400, "Only 1 video can be analyzed at a time.")
            
            try:
                duration = get_video_duration(content)
                if duration > 60:
                    raise HTTPException(400, f"Video is too long ({duration:.1f}s). Maximum allowed is 60 seconds.")
                logger.info(f"[VIDEO] Duration accepted: {duration:.1f}s")
            except Exception as e:
                if isinstance(e, HTTPException): raise
                logger.error(f"Video processing error: {e}")
                raise HTTPException(400, "Could not process video file. Ensure it is a valid format.")

            # Extract frames
            frames = extract_video_frames(content)
            for i, frame_b64 in enumerate(frames):
                # Format as OpenAI-style URL for convert_messages_to_gemini
                visual_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })
                if i == 0:
                    text_parts.append(f"\n[Video Frame Analysis: {filename}]\n")

        # --- IMAGE HANDLING ---
        elif content_type.startswith("image/") or get_file_category(filename) == FileCategory.IMAGE:
            b64 = base64.b64encode(content).decode()
            visual_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            text_parts.append(f"\n[Image Analysis: {filename}]\n")

        # --- TEXT/DOC/CODE HANDLING ---
        else:
            result = await extract_file_content(content, filename)
            text_parts.append(f"\n--- FILE: {filename} ---\n{result.content}")

    # --- ROUTE TO HANDLER ---
    
    # Priority: Visual Analysis (Images/Video)
    if visual_parts:
        logger.info(f"[ANALYSIS] Processing {len(visual_parts)} visual items with prompt: '{prompt[:50]}...'")
        
        # Construct user message: [Prompt, Image, Image, ...]
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        user_content.extend(visual_parts)
        
        messages = [
            {"role": "system", "content": get_system_prompt("") + "\n\nYou are analyzing images and video frames. Be precise."},
            {"role": "user", "content": user_content}
        ]
        
        if stream:
            async def vision_gen():
                task = asyncio.current_task()
                active_streams[user["id"]] = task
                try:
                    async for token in stream_gemini_chat(messages):
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": token})
                    # Note: We don't save to DB in this endpoint unless we add conv_id logic
                    yield sse({"type": "done"})
                except Exception as e:
                    logger.error(f"Vision stream error: {e}")
                    yield sse({"type": "error", "message": str(e)})
                finally:
                    active_streams.pop(user["id"], None)
            return StreamingResponse(vision_gen(), media_type="text/event-stream")
        else:
            reply = await gemini_chat_complete(messages)
            return {"analysis": reply}

    # Fallback: Text Analysis (Code, Docs, Archives)
    if text_parts:
        combined_text = "\n".join(text_parts)
        full_prompt = f"{prompt}\n\n{combined_text}" if prompt else combined_text
        
        messages = [
            {"role": "system", "content": get_system_prompt("") + " Analyze the provided file content."},
            {"role": "user", "content": full_prompt}
        ]
        
        if stream:
            async def text_gen():
                task = asyncio.current_task()
                active_streams[user["id"]] = task
                try:
                    async for token in stream_gemini_chat(messages):
                        if task.cancelled(): break
                        yield sse({"type": "token", "text": token})
                    yield sse({"type": "done"})
                except Exception as e:
                    logger.error(f"Text analysis stream error: {e}")
                    yield sse({"type": "error", "message": str(e)})
                finally:
                    active_streams.pop(user["id"], None)
            return StreamingResponse(text_gen(), media_type="text/event-stream")
        else:
            reply = await gemini_chat_complete(messages)
            return {"analysis": reply}

    raise HTTPException(400, "No valid files provided for analysis.")


@app.get("/file-types")
async def get_supported_file_types():
    return {
        "code": sorted(list(CODE_EXTENSIONS)), "document": sorted(list(DOCUMENT_EXTENSIONS)),
        "data": sorted(list(DATA_EXTENSIONS)), "image": sorted(list(IMAGE_EXTENSIONS)),
        "audio": sorted(list(AUDIO_EXTENSIONS)), "video": sorted(list(VIDEO_EXTENSIONS)),
        "archive": sorted(list(ARCHIVE_EXTENSIONS)),
        "limits": {"max_file_size": format_file_size(MAX_FILE_SIZE)}
    }


# =========================
# SESSION MANAGEMENT
# =========================
@app.post("/session/validate")
async def validate_session(req: Request, res: Response):
    user = await get_user(req, res)
    return {"valid": user.get("session_valid", False), "user_id": user["id"]}

@app.post("/session/refresh")
async def refresh_session(req: Request, res: Response):
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}
    remember = body.get("remember", True)
    user = await get_user(req, res, remember=remember)
    new_token = await create_user_session(user["id"], user.get("fingerprint", ""), remember)
    set_session_cookies(res, user["id"], user.get("fingerprint", ""), new_token, remember)
    return {"status": "refreshed", "user_id": user["id"]}

@app.post("/session/logout")
async def logout(req: Request, res: Response):
    user_id = req.cookies.get(PRIMARY_COOKIE)
    if user_id:
        try: await _execute_supabase_with_retry(supabase.table("user_sessions").update({"is_valid": False}).eq("user_id", user_id), description="Invalidate Sessions")
        except: pass
        if user_id in _session_cache: del _session_cache[user_id]
    clear_session_cookies(res)
    return {"status": "logged_out"}


# =========================
# TTS / STT
# =========================
@app.get("/tts/voices")
async def get_voices(request: Request, response: Response):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.elevenlabs.io/v1/voices", headers=get_elevenlabs_headers())
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"Voices Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch voices.")

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        if not ELEVENLABS_API_KEY: raise HTTPException(500, "ElevenLabs not configured.")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{req.voice}"
        payload = {"text": req.text, "model_id": "eleven_multilingual_v2", "output_format": "mp3_44100_128"}
        async def stream_audio():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=get_elevenlabs_headers(), json=payload) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes(): yield chunk
        return StreamingResponse(stream_audio(), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(500, "TTS failed.")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        if not ELEVENLABS_API_KEY: raise HTTPException(500, "ElevenLabs not configured.")
        file_content = await file.read()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post("https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                data={"model_id": "scribe_v1"},
                files={"file": (file.filename, file_content, file.content_type)})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(500, "STT failed.")


# =========================
# REGENERATE & NEW CHAT
# =========================
@app.post("/regenerate")
async def regenerate_response(req: RegenerateRequest):
    if not req.conversation_id: raise HTTPException(400, "Conversation ID required")
    try:
        history = await get_history(req.conversation_id)
        if history and history[-1]["role"] == "assistant": history.pop()
        prompt = None
        for msg in reversed(history):
            if msg["role"] == "user": prompt = msg["content"]; break
        if not prompt: raise HTTPException(400, "No user message found.")

        system_prompt = get_system_prompt(prompt, req.conversation_id)
        full_history = [{"role": "system", "content": system_prompt}] + history
        
        async def event_gen():
            full_text = ""
            async for token in stream_gemini_chat(full_history):
                full_text += token
                yield sse({"type": "token", "text": token})
            await save_message("system", req.conversation_id, "assistant", full_text)
            yield sse({"type": "done"})
        return StreamingResponse(event_gen(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/newchat")
async def create_new_chat():
    return JSONResponse(content={"conversation_id": str(uuid.uuid4())})


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "3.1.0-Ultra",
        "services": {
            "google_gemini_500b": bool(GOOGLE_API_KEY),
            "groq_gemma_27b": bool(GROQ_API_KEY),
            "replicate_flux2_max": bool(REPLICATE_API_TOKEN),
            "replicate_veo_31": bool(REPLICATE_API_TOKEN),
            "replicate_minimax_music": bool(REPLICATE_API_TOKEN),
            "elevenlabs": bool(ELEVENLABS_API_KEY),
            "judge0": bool(JUDGE0_API_KEY),
            "wolfram": bool(WOLFRAM_ALPHA_API_KEY),
            "serpapi": bool(SERPAPI_API_KEY),
        },
        "models": {
            "primary_llm": PRIMARY_LLM_MODEL,
            "fast_llm": FAST_LLM_MODEL,
            "image_gen": REPLICATE_IMAGE_MODEL,
            "video_gen": REPLICATE_VIDEO_MODEL,
            "music_gen": REPLICATE_MUSIC_MODEL,
            "music_cover": REPLICATE_MUSIC_COVER_MODEL,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
