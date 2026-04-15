import os
import io
import json
import re
import time
import uuid
import asyncio
import logging
import hashlib
import base64
import tempfile
import zipfile
import shutil
import mimetypes
import secrets
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends, Cookie, Response
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

# Document processing
import fitz  # PyMuPDF
import docx

# ---------- ENV KEYS ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")
    
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# MODELS
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() if os.getenv("GROQ_API_KEY") else ""
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL")
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

# ---------- FILE HANDLING CONFIG ----------
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ZIP_SIZE = 200 * 1024 * 1024  # 200MB for zips
MAX_ZIP_ENTRIES = 1000
MAX_EXTRACTED_SIZE = 500 * 1024 * 1024  # 500MB total extracted
MAX_TEXT_LENGTH = 1000000  # 1M characters
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# ---------- AUTH CONFIG ----------
SESSION_DURATION = 365 * 24 * 60 * 60  # 1 year
REFRESH_THRESHOLD = 7 * 24 * 60 * 60  # 7 days
SESSION_CACHE_TTL = 300  # 5 minutes

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("heloxai-server")

app = FastAPI(title="HeloXAI Ultimate Server", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SSE HELPER ----------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ---------------- MODELS ----------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile") 
CODE_MODEL = os.getenv("CODE_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldYLocks",
    "age": 17,
    "country": "England",
    "projects": ["MZ", "LS", "SX", "CB"],
    "socials": {"discord": "@nexisphere123_89431", "twitter": "@NexiSphere"},
}

JUDGE0_LANGUAGES = {
    "c": 50, "cpp": 54, "java": 62, "python": 71, "javascript": 63,
    "typescript": 74, "go": 60, "rust": 73, "csharp": 51, "php": 68,
    "ruby": 72, "swift": 83, "kotlin": 78, "bash": 46, "sql": 82,
}
JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
JUDGE0_KEY = os.getenv("JUDGE0_API_KEY")

if not JUDGE0_KEY:
    logger.warning("Code execution disabled (missing Judge0 API key)")

# ---------- HTTP Client ----------
groq_client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))

# ---------- Active Streams ----------
active_streams: Dict[str, asyncio.Event] = {}
active_stream_lock = asyncio.Lock()

# ---------- Session Cache ----------
_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_last_cleanup = time.time()

async def register_stream(stream_id: str) -> asyncio.Event:
    async with active_stream_lock:
        event = asyncio.Event()
        active_streams[stream_id] = event
        return event

async def unregister_stream(stream_id: str):
    async with active_stream_lock:
        active_streams.pop(stream_id, None)

async def stop_stream(stream_id: str):
    async with active_stream_lock:
        event = active_streams.get(stream_id)
        if event:
            event.set()


# =========================
# FILE TYPE SYSTEM
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
    '.py', '.pyw', '.pyx', '.pyi', '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.vue', '.svelte', '.astro',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',
    '.cs', '.go', '.rs', '.php', '.phtml', '.rb', '.erb', '.swift', '.dart',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    '.lua', '.pl', '.pm', '.r', '.sql', '.graphql', '.gql', '.tf', '.hcl',
    '.sol', '.move', '.cairo', '.toml', '.dockerfile', '.makefile',
    '.json', '.yaml', '.yml', '.xml', '.ini', '.cfg', '.conf', '.env',
    '.md', '.rst', '.tex', '.latex', '.proto',
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
    '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z', '.rar', '.zst', '.lz4',
}

def get_file_category(filename: str) -> FileCategory:
    if not filename:
        return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS:
        return FileCategory.CODE
    elif ext in DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT
    elif ext in DATA_EXTENSIONS:
        return FileCategory.DATA
    elif ext in IMAGE_EXTENSIONS:
        return FileCategory.IMAGE
    elif ext in AUDIO_EXTENSIONS:
        return FileCategory.AUDIO
    elif ext in VIDEO_EXTENSIONS:
        return FileCategory.VIDEO
    elif ext in ARCHIVE_EXTENSIONS:
        return FileCategory.ARCHIVE
    else:
        return FileCategory.UNKNOWN

def get_file_language(filename: str) -> Optional[str]:
    ext_lang_map = {
        '.py': 'python', '.js': 'javascript', '.jsx': 'javascript', '.ts': 'typescript',
        '.tsx': 'typescript', '.html': 'html', '.css': 'css', '.scss': 'scss',
        '.vue': 'vue', '.svelte': 'svelte', '.java': 'java', '.kt': 'kotlin',
        '.scala': 'scala', '.c': 'c', '.cpp': 'cpp', '.cs': 'csharp',
        '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
        '.swift': 'swift', '.dart': 'dart', '.sh': 'bash', '.sql': 'sql',
        '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.xml': 'xml',
        '.md': 'markdown', '.rust': 'rust', '.lua': 'lua', '.r': 'r',
        '.graphql': 'graphql', '.tf': 'hcl', '.hcl': 'hcl', '.toml': 'toml',
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)

def is_binary_file(filename: str, content: bytes = None) -> bool:
    ext = Path(filename).suffix.lower()
    binary_exts = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | ARCHIVE_EXTENSIONS | {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.pyc', '.pyo',
        '.class', '.o', '.obj', '.a', '.lib', '.pdf', '.doc', '.docx',
        '.xls', '.xlsx', '.ppt', '.pptx', '.sqlite', '.db', '.sqlite3',
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp3', '.mp4',
        '.woff', '.woff2', '.ttf', '.otf', '.eot', '.pak', '.bundle',
    }
    if ext in binary_exts:
        return True
    if content and len(content) > 0:
        check_bytes = content[:8192]
        if b'\x00' in check_bytes:
            return True
    return False

def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# =========================
# FILE EXTRACTION SYSTEM
# =========================
class FileExtractionResult:
    def __init__(
        self,
        content: str,
        files: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
        truncated: bool = False,
        original_size: int = 0
    ):
        self.content = content
        self.files = files or []
        self.metadata = metadata or {}
        self.truncated = truncated
        self.original_size = original_size

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "files": self.files,
            "metadata": self.metadata,
            "truncated": self.truncated,
            "original_size": self.original_size
        }

def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
    for encoding in encodings:
        try:
            text = content.decode(encoding, errors='strict' if encoding != 'latin-1' else 'ignore')
            truncated = len(text) > max_length
            if truncated:
                text = text[:max_length] + "\n\n[... Content truncated ...]"
            return text, truncated
        except (UnicodeDecodeError, LookupError):
            continue
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated

async def extract_file_content(
    content: bytes,
    filename: str,
    max_length: int = MAX_TEXT_LENGTH
) -> FileExtractionResult:
    original_size = len(content)
    category = get_file_category(filename)
    metadata = {
        "filename": filename,
        "category": category.value,
        "size": original_size,
        "size_formatted": format_file_size(original_size),
        "language": get_file_language(filename),
    }

    try:
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)

        if category == FileCategory.IMAGE:
            return FileExtractionResult(
                content=f"[Image file: {filename} ({format_file_size(original_size)}) - Use image analysis for visual content]",
                metadata=metadata,
                original_size=original_size
            )

        if category in (FileCategory.AUDIO, FileCategory.VIDEO):
            return FileExtractionResult(
                content=f"[{category.value.capitalize()} file: {filename} ({format_file_size(original_size)}) - Media file]",
                metadata=metadata,
                original_size=original_size
            )

        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)

        if is_binary_file(filename, content):
            return FileExtractionResult(
                content=f"[Binary file: {filename} ({format_file_size(original_size)}) - Cannot extract text]",
                metadata=metadata,
                original_size=original_size
            )

        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(
            content=text,
            metadata=metadata,
            truncated=truncated,
            original_size=original_size
        )

    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return FileExtractionResult(
            content=f"[Error extracting {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=original_size
        )

async def extract_pdf_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    try:
        doc_pdf = fitz.open(stream=content, filetype="pdf")
        pages = []
        for i, page in enumerate(doc_pdf):
            page_text = page.get_text() or ""
            pages.append(f"--- Page {i + 1} ---\n{page_text}")
        full_text = "\n\n".join(pages)
        metadata["page_count"] = len(doc_pdf)
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(
            content=full_text,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
    except Exception as e:
        return FileExtractionResult(
            content=f"[PDF parsing error: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )

async def extract_archive_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    ext = Path(filename).suffix.lower()
    
    if ext == '.zip':
        return await extract_zip_content(content, filename, max_length, metadata)
    elif ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
        return await extract_tar_content(content, filename, max_length, metadata)
    else:
        return FileExtractionResult(
            content=f"[Archive format {ext} not supported for extraction]",
            metadata=metadata,
            original_size=len(content)
        )

async def extract_zip_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    extracted_files = []
    all_text_parts = []
    total_extracted = 0
    entry_count = 0
    
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[ZIP: {filename} - Too many entries ({len(zf.namelist())})]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            entries = sorted(zf.namelist())
            
            for entry_name in entries:
                if entry_name.endswith('/') or '/__MACOSX/' in entry_name:
                    continue
                if entry_name.startswith('__MACOSX') or entry_name.startswith('.'):
                    continue
                
                entry_count += 1
                
                try:
                    entry_info = zf.getinfo(entry_name)
                    
                    if entry_info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": f"File too large (max {format_file_size(MAX_FILE_SIZE)})"
                        })
                        continue
                    
                    if total_extracted + entry_info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": "Archive total size limit reached"
                        })
                        continue
                    
                    entry_content = zf.read(entry_name)
                    total_extracted += len(entry_content)
                    
                    entry_category = get_file_category(entry_name)
                    entry_language = get_file_language(entry_name)
                    
                    if entry_category in (FileCategory.IMAGE, FileCategory.AUDIO, FileCategory.VIDEO):
                        extracted_files.append({
                            "name": entry_name,
                            "size": len(entry_content),
                            "size_formatted": format_file_size(len(entry_content)),
                            "category": entry_category.value,
                            "status": "media",
                        })
                    elif is_binary_file(entry_name, entry_content):
                        extracted_files.append({
                            "name": entry_name,
                            "size": len(entry_content),
                            "size_formatted": format_file_size(len(entry_content)),
                            "category": "binary",
                            "status": "binary",
                        })
                    else:
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        
                        if text.strip():
                            extracted_files.append({
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "language": entry_language,
                                "status": "extracted",
                                "line_count": text.count('\n') + 1,
                                "preview": text[:500] + ("..." if len(text) > 500 else "")
                            })
                            all_text_parts.append(f"\n{'='*60}\nFile: {entry_name}\n{'='*60}\n{text}")
                        else:
                            extracted_files.append({
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "status": "empty",
                            })
                    
                    extracted_files[-1]["name"] = entry_name
                    
                except Exception as e:
                    extracted_files.append({
                        "name": entry_name,
                        "status": "error",
                        "error": str(e)
                    })
        
        full_text = f"ZIP Archive: {filename}\n"
        full_text += f"Total entries: {len(zf.namelist())}, Processed: {entry_count}\n"
        full_text += f"Extracted text files: {len(all_text_parts)}\n"
        full_text += f"Total extracted size: {format_file_size(total_extracted)}\n\n"
        
        if all_text_parts:
            full_text += "="*60 + "\nEXTRACTED CONTENT\n" + "="*60
            full_text += "".join(all_text_parts)
        else:
            full_text += "No text content could be extracted.\n\nFiles found:\n"
            for f in extracted_files:
                status = f.get('status', 'unknown')
                full_text += f"  - {f['name']} ({f.get('size_formatted', '?')}) [{status}]\n"
        
        metadata.update({
            "archive_type": "zip",
            "entry_count": len(zf.namelist()),
            "processed_count": entry_count,
            "extracted_count": len(all_text_parts),
            "total_extracted_size": total_extracted,
            "files": extracted_files
        })
        
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        
        return FileExtractionResult(
            content=full_text,
            files=extracted_files,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
        
    except zipfile.BadZipFile:
        return FileExtractionResult(
            content=f"[Error: {filename} is not a valid ZIP file]",
            metadata=metadata,
            original_size=len(content)
        )
    except Exception as e:
        return FileExtractionResult(
            content=f"[Error extracting ZIP {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )

async def extract_tar_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    import tarfile
    extracted_files = []
    all_text_parts = []
    
    try:
        with tarfile.open(fileobj=io.BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            
            if len(members) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[TAR: {filename} - Too many entries ({len(members)})]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            for member in members:
                if member.name.startswith('./'):
                    member.name = member.name.lstrip('./')
                if member.name.startswith('__MACOSX') or member.name.startswith('.'):
                    continue
                
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    entry_content = f.read()
                    entry_category = get_file_category(member.name)
                    
                    if not is_binary_file(member.name, entry_content):
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        if text.strip():
                            all_text_parts.append(f"\n{'='*60}\nFile: {member.name}\n{'='*60}\n{text}")
                            extracted_files.append({
                                "name": member.name,
                                "size": member.size,
                                "status": "extracted",
                                "category": entry_category.value
                            })
                    else:
                        extracted_files.append({
                            "name": member.name,
                            "size": member.size,
                            "status": "binary",
                            "category": entry_category.value
                        })
                except Exception as e:
                    extracted_files.append({"name": member.name, "status": "error", "error": str(e)})
        
        full_text = f"TAR Archive: {filename}\nEntries: {len(members)}, Extracted: {len(all_text_parts)}\n\n"
        if all_text_parts:
            full_text += "".join(all_text_parts)
        
        metadata.update({
            "archive_type": "tar",
            "entry_count": len(members),
            "extracted_count": len(all_text_parts),
            "files": extracted_files
        })
        
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        
        return FileExtractionResult(
            content=full_text,
            files=extracted_files,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
    except Exception as e:
        return FileExtractionResult(
            content=f"[Error extracting TAR {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )


# =========================
# PRODUCTION-GRADE AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloX_Session"
FINGERPRINT_COOKIE = "HeloX_FP"
BACKUP_COOKIE = "HeloX_ID"
DEVICE_COOKIE = "HeloX_Dev"
SESSION_TOKEN_COOKIE = "HeloX_Token"
SESSION_EXPIRY_COOKIE = "HeloX_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    if remember:
        return {"max_age": SESSION_DURATION, "httponly": True, "secure": True, "samesite": "none", "path": "/"}
    else:
        return {"max_age": 24 * 60 * 60, "httponly": True, "secure": True, "samesite": "none", "path": "/"}

def generate_device_fingerprint(request: Request) -> str:
    fp_components = [
        request.headers.get("user-agent", ""),
        request.headers.get("accept-language", ""),
        request.headers.get("accept-encoding", ""),
        request.headers.get("sec-ch-ua-platform", ""),
        request.headers.get("sec-ch-ua-mobile", ""),
        request.client.host if request.client else "",
    ]
    return hashlib.sha256("|".join(fp_components).encode()).hexdigest()[:32]

def generate_session_token() -> str:
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
    try:
        return time.time() > int(expiry_str)
    except:
        return True

def should_refresh_session(expiry_str: str) -> bool:
    try:
        return (int(expiry_str) - time.time()) < REFRESH_THRESHOLD
    except:
        return True

async def validate_session_token(user_id: str, token: str) -> bool:
    try:
        if user_id in _session_cache:
            if _session_cache[user_id].get("token") == token:
                return True
        
        result = await asyncio.to_thread(
            lambda: supabase.table("user_sessions")
            .select("token, expires_at")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {"token": token, "expires_at": result.data[0].get("expires_at")}
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

async def create_user_session(user_id: str, fingerprint: str, remember: bool = True, user_agent: str = "", ip: str = "") -> str:
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_DURATION if remember else 24 * 60 * 60)
    
    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "token": token,
                "fingerprint": fingerprint,
                "user_agent": user_agent[:500] if user_agent else "",
                "ip_address": ip[:45] if ip else "",
                "expires_at": expires_at.isoformat(),
                "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        )
        _session_cache[user_id] = {"token": token, "expires_at": expires_at.isoformat()}
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
    
    return token

async def cleanup_session_cache():
    global _session_cache_last_cleanup
    now = time.time()
    if now - _session_cache_last_cleanup < SESSION_CACHE_TTL:
        return
    _session_cache_last_cleanup = now
    expired = [uid for uid, data in _session_cache.items() if is_session_expired(data.get("expires_at", "0"))]
    for uid in expired:
        del _session_cache[uid]

async def get_current_user_optional(
    request: Request,
    response: Response,
    HeloX_Session: Optional[str] = Cookie(None),
    HeloX_Token: Optional[str] = Cookie(None),
    HeloX_Expiry: Optional[str] = Cookie(None),
    HeloX_FP: Optional[str] = Cookie(None),
    HeloX_ID: Optional[str] = Cookie(None),
    HeloX_Dev: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    await cleanup_session_cache()
    
    current_fingerprint = generate_device_fingerprint(request)
    user_agent = request.headers.get("user-agent", "")
    client_ip = request.client.host if request.client else ""
    
    # Check Bearer token first
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        return {"id": token, "is_authenticated": True}
    
    user_id = None
    session_valid = False
    remember = True
    
    # Validate existing session
    if HeloX_Session and HeloX_Token:
        if is_session_expired(HeloX_Expiry or "0"):
            clear_session_cookies(response)
        else:
            if await validate_session_token(HeloX_Session, HeloX_Token):
                user_id = HeloX_Session
                session_valid = True
                remember = not is_session_expired(HeloX_Expiry or "0")
    
    # Fallback to backup cookie
    if not user_id and HeloX_ID:
        user_id = HeloX_ID
    
    # Fallback to device fingerprint lookup
    if not user_id and HeloX_Dev:
        fp_part = HeloX_Dev.split("_")[0] if "_" in HeloX_Dev else HeloX_Dev
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1).execute()
            )
            if result.data:
                user_id = result.data[0]["id"]
        except:
            pass
    
    # Fallback to stored fingerprint
    if not user_id and HeloX_FP:
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("users").select("id").eq("fingerprint", HeloX_FP).limit(1).execute()
            )
            if result.data:
                user_id = result.data[0]["id"]
        except:
            pass
    
    # Load user data
    if user_id:
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("users").select("*").eq("id", user_id).limit(1).execute()
            )
            if result.data:
                u = result.data[0]
                
                # Update fingerprint if changed
                if u.get("fingerprint") != current_fingerprint:
                    try:
                        await asyncio.to_thread(
                            lambda: supabase.table("users").update({"fingerprint": current_fingerprint}).eq("id", user_id).execute()
                        )
                    except:
                        pass
                
                # Create/refresh session if invalid
                session_token = HeloX_Token
                if not session_valid:
                    session_token = await create_user_session(user_id, current_fingerprint, remember, user_agent, client_ip)
                    session_valid = True
                
                set_session_cookies(response, user_id, current_fingerprint, session_token, remember)
                
                return {
                    "id": u["id"],
                    "email": u.get("email"),
                    "is_premium": u.get("is_premium", False),
                    "is_lifetime": u.get("is_lifetime", False),
                    "plan": u.get("plan", "free"),
                    "session_valid": session_valid,
                    "fingerprint": current_fingerprint
                }
        except Exception as e:
            logger.error(f"User data fetch failed: {e}")
    
    # Create new anonymous user
    new_id = str(uuid.uuid4())
    try:
        await asyncio.to_thread(
            lambda: supabase.table("users").upsert({
                "id": new_id,
                "email": f"anon+{new_id[:8]}@local",
                "fingerprint": current_fingerprint
            }, on_conflict="id").execute()
        )
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
    
    session_token = await create_user_session(new_id, current_fingerprint, True, user_agent, client_ip)
    set_session_cookies(response, new_id, current_fingerprint, session_token, True)
    
    return {"id": new_id, "session_valid": True, "fingerprint": current_fingerprint}


# =========================
# ADVANCED INTENT DETECTION
# =========================
class IntentCategory(Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    CODE_EXECUTION = "code_execution"
    DOCUMENT_CREATION = "document_creation"
    DOCUMENT_ANALYSIS = "document_analysis"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    WEB_DEVELOPMENT = "web_development"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    WEB_SEARCH = "web_search"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VISION_ANALYSIS = "vision_analysis"
    JOKE = "joke"
    CONVERSATION = "conversation"

@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    sub_intents: List[IntentCategory]
    keywords_matched: List[str]
    patterns_matched: List[str]
    legacy_intent: str

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "sub_intents": [i.value for i in self.sub_intents],
            "keywords_matched": self.keywords_matched,
            "patterns_matched": self.patterns_matched,
            "legacy_intent": self.legacy_intent
        }

class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {"don't", "dont", "do not", "doesn't", "doesnt", "does not", "didn't", "didnt", "did not", "never", "no", "not", "without", "skip", "avoid", "except", "but not", "ignore", "rather than"}
        self.legacy_intent_map = {
            IntentCategory.IMAGE_GENERATION: "image_generation", IntentCategory.VIDEO_GENERATION: "video_generation",
            IntentCategory.AUDIO_GENERATION: "audio_generation", IntentCategory.CODE_GENERATION: "code_generation",
            IntentCategory.CODE_REVIEW: "code_generation", IntentCategory.CODE_DEBUG: "code_generation",
            IntentCategory.CODE_EXECUTION: "code_execution", IntentCategory.DOCUMENT_CREATION: "document_creation",
            IntentCategory.DOCUMENT_ANALYSIS: "document_creation", IntentCategory.DATA_ANALYSIS: "data_analysis",
            IntentCategory.DATA_VISUALIZATION: "data_analysis", IntentCategory.WEB_DEVELOPMENT: "code_generation",
            IntentCategory.API_DEVELOPMENT: "code_generation", IntentCategory.DATABASE: "code_generation",
            IntentCategory.TRANSLATION: "translation", IntentCategory.SUMMARIZATION: "summarization",
            IntentCategory.EXPLANATION: "reasoning", IntentCategory.REASONING: "reasoning",
            IntentCategory.CREATIVE_WRITING: "creative_writing", IntentCategory.MATHEMATICAL: "math_calculation",
            IntentCategory.RESEARCH: "web_search", IntentCategory.WEB_SEARCH: "web_search",
            IntentCategory.TEXT_TO_SPEECH: "text_to_speech", IntentCategory.AUDIO_TRANSCRIPTION: "audio_transcription",
            IntentCategory.VISION_ANALYSIS: "vision_analysis", IntentCategory.JOKE: "joke", IntentCategory.CONVERSATION: "chat",
        }

    def _compile_patterns(self):
        self.patterns = {
            IntentCategory.IMAGE_GENERATION: [r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)', r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)', r'\b(text\s+to\s+image|txt2img|img2img)', r'\b(visualize|visualise)\s+(this|that|the|it)', r'\b(dall[eé]|midjourney|stable\s+diffusion|sd\s*xl|flux)', r'\b(generate|create)\s+(some\s+)?art', r'\bmake\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster)'],
            IntentCategory.VIDEO_GENERATION: [r'\b(generate|create|make|produce)\s+(a\s+)?(video|clip|movie|animation|motion\s+graphic)', r'\b(text\s+to\s+video|txt2vid|video\s+generation)', r'\b(animate|animation)\s+(this|that|the|image|picture)', r'\b(video|clip|movie)\s+(of|showing|about|with)', r'\b(runway|pika|sora|mov2mov|kling|gen-3)', r'\b(turn|convert)\s+(this|the|image)\s+(into|to)\s+(a\s+)?(video|animation)'],
            IntentCategory.AUDIO_GENERATION: [r'\b(generate|create|make|produce)\s+(a\s+)?(audio|sound|music|speech|voice|song|track|beat)', r'\b(text\s+to\s+speech|tts|speech\s+to\s+text|stt)', r'\b(music|song|beat|melody)\s+(generation|creation|for|about)', r'\b(elevenlabs|suno|udio|bark)', r'\b(clone|replicate)\s+(a\s+)?voice'],
            IntentCategory.CODE_GENERATION: [r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)', r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))', r'\b(code\s+(for|that|this|to|which|example))', r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))', r'\b(scaffold|boilerplate|template)\s+(for|a)', r'\b(implement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature))'],
            IntentCategory.CODE_REVIEW: [r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)', r'\b(is\s+(this|there)\s+(code|anything)\s+(good|bad|wrong|improvable|clean))', r'\b(best\s+practices?\s+(for|in)\s+(this|my)\s+(code|implementation))', r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)'],
            IntentCategory.CODE_DEBUG: [r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)', r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring|returning))', r'\b(error|exception|traceback|stack\s+trace|segfault)\s*[:\n]', r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)'],
            IntentCategory.CODE_EXECUTION: [r'\b(run|execute)\s+(this|my|the)\s+(code|script|program)', r'\b(output|result)\s+(of|for)\s+(this|the)\s+(code|program)', r'\b(test\s+(this|my|the)\s+(code|function|program))'],
            IntentCategory.DOCUMENT_CREATION: [r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)', r'\b(professional|formal|business)\s+(document|letter|email|report)'],
            IntentCategory.DOCUMENT_ANALYSIS: [r'\b(analyze|analysis|analyse|read|parse|extract)\s+(this|the|my)\s+(document|pdf|file|doc|report)', r'\b(summarize\s+(this|the)\s+(document|pdf|file|report))'],
            IntentCategory.DATA_ANALYSIS: [r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)', r'\b(statistics?|statistical)\s+(analysis|test|summary|overview)', r'\b(insights?\s+(from|in|about|into))'],
            IntentCategory.DATA_VISUALIZATION: [r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)', r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap)'],
            IntentCategory.WEB_DEVELOPMENT: [r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page|web\s*site|portfolio)', r'\b(html|css|javascript|typescript|react|vue|angular|next\.js|nuxt|svelte|tailwind)\b'],
            IntentCategory.API_DEVELOPMENT: [r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)', r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service|endpoint)?'],
            IntentCategory.DATABASE: [r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)', r'\b(sql|mysql|postgres|postgresql|mongodb|redis|dynamodb|sqlite)\s*(query|statement|command)?'],
            IntentCategory.TRANSLATION: [r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)', r'\b(in\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu))'],
            IntentCategory.SUMMARIZATION: [r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)', r'\b(key\s+(points|takeaways|highlights))\s*(from|of|in)?'],
            IntentCategory.EXPLANATION: [r'\b(explain|explanation)\s+(to\s+me\s+)?', r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+', r'\b(how\s+(does|do|did|can|would|should|to))\s+'],
            IntentCategory.REASONING: [r'\b(reason|think)\s+(through|step\s+by\s+step)', r'\b(step\s+by\s+step)', r'\b(logical|critical|analytical)\s+(thinking|reasoning|analysis)'],
            IntentCategory.CREATIVE_WRITING: [r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|song|haiku|limerick)', r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller|mystery)\s*(writing|story|tale)?'],
            IntentCategory.MATHEMATICAL: [r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem|integral|derivative)?', r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability|linear\s+algebra)\s*(problem|equation|question)?'],
            IntentCategory.RESEARCH: [r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)', r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper|article|research|journal)?'],
            IntentCategory.WEB_SEARCH: [r'\b(search|google)\s+(for|about|on)\s+', r'\b(what\s+is|who\s+is|where\s+is|when\s+(is|was|did))\s+', r'\b(latest|current|recent|news|update[s]?)\s+(on|about|for)\s+'],
            IntentCategory.TEXT_TO_SPEECH: [r'\b(speak|say|read|narrate|pronounce)\s+(this|it|the|that)\s+', r'\b(text\s+to\s+speech|tts)\b'],
            IntentCategory.AUDIO_TRANSCRIPTION: [r'\b(transcribe\s+(this\s+)?(audio|voice|recording|speech))', r'\b(speech\s+to\s+text|stt)\b'],
            IntentCategory.VISION_ANALYSIS: [r'\b(analyze|describe|what(?:\'s| is)\s+in)\s+(this\s+)?(image|picture|photo)', r'\b(what\s+(do\s+you\s+)?see\s+in)'],
            IntentCategory.JOKE: [r'\btell\s+(me\s+)?(a\s+)?joke', r'\bmake\s+me\s+laugh', r'\bsomething\s+funny'],
            IntentCategory.CONVERSATION: [r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$', r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$', r'^(how\s+are\s+you|how(\'s|\s+is)\s+it\s+going|what(\'s|\s+is)\s+up)[\s!.?]*$', r'^(bye|goodbye|see\s+you|farewell)[\s!.?]*$'],
        }
        self.compiled_patterns = {intent: [re.compile(p, re.IGNORECASE) for p in patterns] for intent, patterns in self.patterns.items()}

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.IMAGE_GENERATION: ["image", "picture", "photo", "drawing", "illustration", "artwork", "painting", "sketch", "graphic", "visual", "render", "thumbnail", "logo", "icon", "banner", "poster", "dalle", "midjourney", "stable diffusion", "ai art"],
            IntentCategory.VIDEO_GENERATION: ["video", "clip", "movie", "film", "animation", "motion", "gif", "runway", "pika", "sora"],
            IntentCategory.AUDIO_GENERATION: ["audio", "sound", "music", "speech", "voice", "song", "track", "beat", "tts", "elevenlabs", "suno"],
            IntentCategory.CODE_GENERATION: ["code", "script", "function", "class", "module", "program", "app", "application", "snippet", "algorithm"],
            IntentCategory.CODE_DEBUG: ["bug", "error", "issue", "problem", "debug", "fix", "troubleshoot", "exception", "crash", "broken", "wrong"],
            IntentCategory.MATHEMATICAL: ["calculate", "compute", "solve", "math", "equation", "formula", "integral", "derivative", "proof"],
            IntentCategory.RESEARCH: ["research", "find", "search", "investigate", "study", "academic", "citation", "reference"],
        }

    def _has_negation(self, text: str, keyword_pos: int) -> bool:
        words_before = text[:keyword_pos].lower().split()[-6:]
        return any(neg in " ".join(words_before) for neg in self.negation_words)

    def _calculate_confidence(self, matched_keywords, matched_patterns, text_length) -> float:
        if not matched_keywords and not matched_patterns: return 0.0
        return min((min(len(matched_patterns) * 0.35, 0.65) + min(len(matched_keywords) * 0.12, 0.25) + (0.1 if matched_keywords and matched_patterns else 0.0)) * max(0.5, 1.0 - (text_length / 1500) * 0.4), 1.0)

    def _are_related_intents(self, i1: IntentCategory, i2: IntentCategory) -> bool:
        groups = [{IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG, IntentCategory.CODE_EXECUTION}, {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION}, {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION}, {IntentCategory.RESEARCH, IntentCategory.WEB_SEARCH}]
        return any(i1 in g and i2 in g for g in groups)

    def detect_intents(self, text: str, threshold: float = 0.25) -> List[IntentResult]:
        text_lower = text.lower()
        results = []
        for intent, patterns in self.compiled_patterns.items():
            matched_keywords, matched_patterns = [], []
            for p in patterns:
                if p.search(text): matched_patterns.append(p.pattern)
            for syn in self.synonyms.get(intent, []):
                if syn in text_lower:
                    pos = text_lower.find(syn)
                    if not self._has_negation(text, pos): matched_keywords.append(syn)
            if matched_keywords or matched_patterns:
                conf = self._calculate_confidence(matched_keywords, matched_patterns, len(text))
                if conf >= threshold:
                    results.append(IntentResult(intent=intent, confidence=conf, sub_intents=[], keywords_matched=matched_keywords, patterns_matched=matched_patterns, legacy_intent=self.legacy_intent_map.get(intent, "chat")))
        results.sort(key=lambda x: x.confidence, reverse=True)
        if results:
            for r in results[1:]:
                if self._are_related_intents(results[0].intent, r.intent): results[0].sub_intents.append(r.intent)
        return results[:1] if results else []

    def get_primary_intent(self, text: str) -> Optional[IntentResult]:
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> str:
        intent = self.get_primary_intent(text)
        return intent.legacy_intent if intent else "chat"

    def get_required_tools(self, text: str) -> List[str]:
        intent = self.get_primary_intent(text)
        if not intent: return ["llm"]
        tool_map = {IntentCategory.IMAGE_GENERATION: ["image_gen", "llm"], IntentCategory.VIDEO_GENERATION: ["video_gen", "llm"], IntentCategory.AUDIO_GENERATION: ["audio_gen", "llm"], IntentCategory.CODE_GENERATION: ["code_gen", "llm"], IntentCategory.CODE_EXECUTION: ["code_exec", "llm"], IntentCategory.DATA_ANALYSIS: ["code_exec", "data_processing", "llm"], IntentCategory.MATHEMATICAL: ["wolfram", "code_exec", "llm"], IntentCategory.RESEARCH: ["web_search", "llm"], IntentCategory.WEB_SEARCH: ["web_search", "llm"], IntentCategory.TEXT_TO_SPEECH: ["tts", "llm"], IntentCategory.VISION_ANALYSIS: ["vision", "llm"]}
        return list(tool_map.get(intent.intent, ["llm"]))

    def get_code_system_prompt(self, text: str) -> str:
        intent = self.get_primary_intent(text)
        prompts = {IntentCategory.CODE_DEBUG: "Expert debugger. Identify root cause, explain WHY, provide exact fix.", IntentCategory.CODE_REVIEW: "Senior code reviewer. Check quality, bugs, security, performance.", IntentCategory.CODE_GENERATION: "Generate clean, well-documented code. Return ONLY code in a code block."}
        return prompts.get(intent.intent if intent else None, "Generate clean, well-documented code.")

_detector = None
def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None: _detector = AdvancedIntentDetector()
    return _detector

def detect_intent(prompt: str) -> tuple:
    if not prompt: return ("chat", 0.0)
    result = get_detector().get_primary_intent(prompt)
    return (result.legacy_intent, result.confidence) if result else ("chat", 0.0)

def detect_intent_advanced(prompt: str) -> Optional[IntentResult]:
    return get_detector().get_primary_intent(prompt)

def get_required_tools(prompt: str) -> List[str]:
    return get_detector().get_required_tools(prompt)


# ---------- Request/Response Models ----------
class UniversalRequest(BaseModel):
    prompt: str = Field(..., description="The prompt or query")
    context: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    tools: Optional[List[str]] = None
    files: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------- Helper Functions ----------
def get_groq_headers() -> dict:
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def detect_language(prompt: str) -> str:
    lang_map = {"python": ["python", "py", "django", "flask"], "javascript": ["javascript", "js", "node", "react", "typescript", "ts"], "java": ["java", "spring"], "cpp": ["c++", "cpp"], "go": ["golang", "go lang"], "rust": ["rust", "rustlang"], "sql": ["sql", "mysql", "postgres"]}
    for lang, keywords in lang_map.items():
        if any(k in prompt.lower() for k in keywords): return lang
    return "python"

def get_elevenlabs_voice_id(voice_name: str) -> str:
    return {"alloy": "21m00Tcm4TlvDq8ikWAM", "rachel": "21m00Tcm4TlvDq8ikWAM", "adam": "pNInz6obpgDQGcFmaJgB", "antoni": "ErXwobaYiN019PkySvjV", "bella": "EXAVITQu4vr4xnSDxMaL"}.get(voice_name.lower(), "21m00Tcm4TlvDq8ikWAM")

async def extract_document_text(doc: str) -> str:
    try:
        if doc.startswith("http"):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(doc)
                resp.raise_for_status()
                data = resp.content
        else:
            if "," in doc: doc = doc.split(",", 1)[1]
            data = base64.b64decode(doc)
        try:
            doc_pdf = fitz.open(stream=data, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc_pdf])
            if text.strip(): return text
        except: pass
        try:
            doc_docx = docx.Document(io.BytesIO(data))
            text = "\n".join([p.text for p in doc_docx.paragraphs])
            if text.strip(): return text
        except: pass
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error: {str(e)}]"

async def serper_search(query: str) -> list:
    if not SERPER_API_KEY: return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post("https://google.serper.dev/search", headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}, data=json.dumps({"q": query, "num": 10}))
            data = response.json()
            results = []
            if data.get("knowledgeGraph"):
                kg = data["knowledgeGraph"]
                results.append({"title": kg.get("title", "Summary"), "snippet": kg.get("description", ""), "url": kg.get("descriptionLink", ""), "type": "knowledge_graph"})
            for item in data.get("organic", [])[:8]:
                results.append({"title": item.get("title"), "snippet": item.get("snippet"), "url": item.get("link"), "type": "organic"})
            return results
    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return []

async def tell_joke(category: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": f"Tell a funny {category} joke. Return only the joke."}], "max_tokens": 200, "temperature": 0.9})
            return {"joke": r.json()["choices"][0]["message"]["content"]}
    except: return {"joke": "Why don't scientists trust atoms? Because they make up everything!"}

async def solve_math(prompt: str) -> dict:
    if WOLFRAM_ALPHA_API_KEY:
        try:
            import wolframalpha
            client = wolframalpha.Client(WOLFRAM_ALPHA_API_KEY)
            res = client.query(prompt)
            return {"answer": next(res.results).text, "method": "wolfram"}
        except: pass
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": f"Solve step by step:\n{prompt}"}], "max_tokens": 1000, "temperature": 0.2})
            return {"answer": r.json()["choices"][0]["message"]["content"], "method": "llm"}
    except: return {"answer": "Unable to solve", "method": "none"}

async def chat_with_tools(user_id: str, messages: list) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": messages, "max_tokens": 4096, "temperature": 0.7})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def fetch_user_memory(user_id: str) -> str:
    if not user_id or user_id == "anonymous": return ""
    try:
        res = await asyncio.to_thread(lambda: supabase.table("user_memories").select("category, content").eq("user_id", user_id).order("last_referenced", desc=True).limit(10).execute())
        if not res.data: return ""
        return "User Profile & Memories:\n" + "\n".join([f"- [{m.get('category', 'info')}] {m['content']}" for m in res.data])
    except: return ""

async def extract_and_save_memory(user_id: str, prompt: str, response: str):
    if not user_id or user_id == "anonymous": return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": f"Extract facts about USER from this. Return JSON array of {{\"category\", \"content\"}} or []\nUser: {prompt[:500]}\nAI: {response[:500]}"}], "temperature": 0.1, "max_tokens": 200})
            raw = r.json()["choices"][0]["message"]["content"]
            if "```json" in raw: raw = raw.split("```json")[1].split("```")[0]
            for fact in json.loads(raw):
                if fact.get("content"):
                    await asyncio.to_thread(lambda f=fact: supabase.table("user_memories").upsert({"user_id": user_id, "category": f.get("category", "info"), "content": f["content"], "last_referenced": datetime.utcnow().isoformat()}, on_conflict="user_id,content").execute())
    except: pass


# =========================
# ENDPOINTS
# =========================
@app.post("/analyze-intent")
async def analyze_intent_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt: raise HTTPException(400, "Prompt required")
    intent_result = detect_intent_advanced(prompt)
    return {"intent": intent_result.to_dict() if intent_result else None, "legacy_intent": intent_result.legacy_intent if intent_result else "chat", "confidence": intent_result.confidence if intent_result else 0.0, "required_tools": get_required_tools(prompt)}

@app.get("/file-types")
async def get_supported_file_types():
    return {
        "code": sorted(list(CODE_EXTENSIONS)), "document": sorted(list(DOCUMENT_EXTENSIONS)),
        "data": sorted(list(DATA_EXTENSIONS)), "image": sorted(list(IMAGE_EXTENSIONS)),
        "audio": sorted(list(AUDIO_EXTENSIONS)), "video": sorted(list(VIDEO_EXTENSIONS)),
        "archive": sorted(list(ARCHIVE_EXTENSIONS)),
        "limits": {"max_file_size": format_file_size(MAX_FILE_SIZE), "max_zip_size": format_file_size(MAX_ZIP_SIZE), "max_text_length": format_file_size(MAX_TEXT_LENGTH)}
    }

@app.post("/analysis")
async def analyze_file(request: Request, file: UploadFile = File(...), stream: bool = True):
    """Enhanced file analysis supporting all file types including archives"""
    user = await get_current_user_optional(request, Response())
    content = await file.read()
    filename = file.filename or "unknown"
    file_size = len(content)
    
    logger.info(f"[FILE] Upload: {filename} ({format_file_size(file_size)})")
    
    if not content: raise HTTPException(400, "Empty file")
    
    category = get_file_category(filename)
    max_allowed = MAX_ZIP_SIZE if category == FileCategory.ARCHIVE else MAX_FILE_SIZE
    
    if file_size > max_allowed:
        raise HTTPException(400, f"File too large ({format_file_size(file_size)}). Max: {format_file_size(max_allowed)}")
    
    if category == FileCategory.IMAGE:
        if not OPENAI_API_KEY: raise HTTPException(500, "OpenAI Key required for image analysis")
        b64 = base64.b64encode(content).decode()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [{"type": "text", "text": "Analyze this image in detail."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
            return {"analysis": r.json()["choices"][0]["message"]["content"], "metadata": {"category": "image", "size": file_size}}
    
    result = await extract_file_content(content, filename)
    
    if category == FileCategory.ARCHIVE and result.files:
        # Special handling for archives
        messages = [{"role": "system", "content": f"You are analyzing an archive. Archive info:\n- Entries: {result.metadata.get('entry_count')}\n- Extracted: {result.metadata.get('extracted_count')}\n\nProvide an overview of what this archive contains."}, {"role": "user", "content": result.content[:50000]}]
    else:
        file_context = f"\nFile: {filename}\nCategory: {result.metadata.get('category')}\nSize: {format_file_size(file_size)}\nLines: {result.metadata.get('line_count', 'N/A')}\n"
        if result.metadata.get("language"):
            file_context += f"Language: {result.metadata['language']}\n"
        messages = [{"role": "system", "content": f"You analyze files. Detect type and respond accordingly:\n- Code → explain, find bugs, suggest improvements\n- Documents → summarize + key insights\n- Data → extract patterns\nBe structured.{file_context}"}, {"role": "user", "content": result.content}]
    
    if stream:
        async def gen():
            try:
                yield sse({"type": "file_metadata", "metadata": result.metadata, "files": result.files})
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": messages, "stream": True, "max_tokens": 4096}) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content: yield sse({"type": "token", "text": content})
                                except: pass
                yield sse({"type": "done"})
            except Exception as e: yield sse({"type": "error", "message": str(e)})
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": messages, "max_tokens": 4096})
        return {"analysis": r.json()["choices"][0]["message"]["content"], "metadata": result.metadata, "files": result.files, "truncated": result.truncated}

@app.post("/ask/universal")
async def ask_universal(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    try:
        content_type = request.headers.get("content-type", "")
        
        # Handle multipart form data with file uploads
        if "multipart/form-data" in content_type:
            form = await request.form()
            prompt = form.get("prompt", "").strip() if form.get("prompt") else ""
            file = form.get("file")
            
            if file:
                file_content = await file.read()
                filename = file.filename or "unknown"
                
                # Route to file analysis
                if file.content_type and file.content_type.startswith("image/"):
                    if OPENAI_API_KEY:
                        b64 = base64.b64encode(file_content).decode()
                        async with httpx.AsyncClient(timeout=60) as client:
                            r = await client.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt or "Analyze this image."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
                            return {"analysis": r.json()["choices"][0]["message"]["content"]}
                
                # Extract file content and analyze
                result = await extract_file_content(file_content, filename)
                messages = [{"role": "system", "content": f"You analyze files. File: {filename}, Category: {result.metadata.get('category')}, Size: {format_file_size(len(file_content))}"}, {"role": "user", "content": (prompt + "\n\n" if prompt else "") + result.content}]
                
                async def gen():
                    try:
                        yield sse({"type": "file_metadata", "metadata": result.metadata})
                        async with httpx.AsyncClient(timeout=None) as client:
                            async with client.stream("POST", GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": messages, "stream": True, "max_tokens": 4096}) as resp:
                                async for line in resp.aiter_lines():
                                    if line.startswith("data: ") and line != "data: [DONE]":
                                        try:
                                            data = json.loads(line[6:])
                                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                            if content: yield sse({"type": "token", "text": content})
                                        except: pass
                        yield sse({"type": "done"})
                    except Exception as e: yield sse({"type": "error", "message": str(e)})
                return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
            
            if not prompt: raise HTTPException(400, "prompt or file required")
        
        body = await request.json()
        prompt = (body.get("prompt") or "").strip()
        conversation_id = body.get("conversation_id")
        files = body.get("files", [])
        stream = body.get("stream", True)
        
        output_type = body.get("output_type", "text")
        language = body.get("language")
        target_language = body.get("target_language")
        voice = body.get("voice", "alloy")
        enable_cot = body.get("enable_cot", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 4096)
        system_prompt = body.get("system_prompt")
        execute_code = body.get("execute_code", False)
        context = body.get("context")
        documents = body.get("documents", [])

        if not prompt and not files and not documents:
            raise HTTPException(status_code=400, detail="prompt, files, or documents required")

        identity = current_user or {}
        user_id = identity.get("id") or str(uuid.uuid4())

        if not conversation_id: conversation_id = str(uuid.uuid4())

        await asyncio.to_thread(lambda: supabase.table("conversations").upsert({"id": conversation_id, "user_id": user_id, "created_at": datetime.utcnow().isoformat()}).execute())

        history_res = await asyncio.to_thread(lambda: supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(20).execute())
        history_messages = history_res.data or []

        intent_result = detect_intent_advanced(prompt)
        detected_intent = intent_result.legacy_intent if intent_result else "chat"
        confidence = intent_result.confidence if intent_result else 0.0
        
        logger.info(f"[INTENT] {detected_intent} ({confidence:.2%}) user={user_id[:8]}")
        
        if output_type == "code": detected_intent = "code_generation"
        elif target_language: detected_intent = "translation"
        elif enable_cot: detected_intent = "reasoning"
        elif execute_code: detected_intent = "code_execution"

        if detected_intent == "joke": return await tell_joke("general")
        elif detected_intent == "math_calculation":
            async def math_gen():
                try:
                    yield sse({"type": "status", "status": "computing"})
                    result = await solve_math(prompt)
                    answer = result["answer"]
                    for char in answer:
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.01)
                    yield sse({"type": "done"})
                except Exception as e: yield sse({"type": "error", "message": str(e)})
            return StreamingResponse(math_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        elif detected_intent == "code_generation":
            lang = language or detect_language(prompt)
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CODE_MODEL, "messages": [{"role": "system", "content": get_detector().get_code_system_prompt(prompt)}, {"role": "user", "content": f"Write {lang} code for: {prompt}"}], "max_tokens": max_tokens, "temperature": temperature})
                return {"language": lang, "code": r.json()["choices"][0]["message"]["content"]}
        elif detected_intent == "code_execution" and JUDGE0_KEY:
            lang = language or detect_language(prompt)
            async def exec_gen():
                try:
                    yield sse({"type": "status", "status": "generating"})
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CODE_MODEL, "messages": [{"role": "user", "content": f"Write executable {lang} code for: {prompt}"}], "max_tokens": max_tokens})
                        code = r.json()["choices"][0]["message"]["content"]
                    match = re.search(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
                    if match: code = match.group(1)
                    yield sse({"type": "code", "language": lang, "code": code})
                    yield sse({"type": "status", "status": "executing"})
                    lang_id = JUDGE0_LANGUAGES.get(lang.lower(), 71)
                    async with httpx.AsyncClient(timeout=30) as client:
                        submit = await client.post(f"{JUDGE0_URL}/submissions", headers={"X-RapidAPI-Key": JUDGE0_KEY, "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com", "Content-Type": "application/json"}, json={"source_code": code, "language_id": lang_id, "stdin": ""})
                        token = submit.json()["token"]
                        for _ in range(15):
                            await asyncio.sleep(1)
                            result = await client.get(f"{JUDGE0_URL}/submissions/{token}", headers={"X-RapidAPI-Key": JUDGE0_KEY, "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"})
                            if result.json().get("status", {}).get("id", 0) in [1, 2]: continue
                            yield sse({"type": "execution", "exit_code": result.json().get("status", {}).get("id"), "output": result.json().get("stdout") or result.json().get("stderr", "No output")})
                            break
                    yield sse({"type": "done"})
                except Exception as e: yield sse({"type": "error", "message": str(e)})
            return StreamingResponse(exec_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        elif detected_intent in ("web_search", "research"):
            query = re.sub(r"^(search for|look up|find|google|research)\s+", "", prompt.lower(), flags=re.IGNORECASE)
            async def search_gen():
                try:
                    yield sse({"type": "status", "status": "searching"})
                    results = await serper_search(query)
                    yield sse({"type": "search_results", "results": results})
                    yield sse({"type": "status", "status": "summarizing"})
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": f"Answer: {query}\nResults: {json.dumps(results)}"}], "max_tokens": 2048, "temperature": 0.3})
                        summary = r.json()["choices"][0]["message"]["content"]
                    for char in summary:
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.01)
                    yield sse({"type": "done"})
                except Exception as e: yield sse({"type": "error", "message": str(e)})
            return StreamingResponse(search_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        elif detected_intent == "text_to_speech":
            text = re.sub(r"[#*`\[\]]", "", prompt).strip()
            text = re.sub(r"^(speak|say|read|narrate)\s+(this|it|the|that)?\s*", "", text, flags=re.IGNORECASE)
            if ELEVENLABS_API_KEY:
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(f"https://api.elevenlabs.io/v1/text-to-speech/{get_elevenlabs_voice_id(voice)}", headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}, json={"text": text, "model_id": "eleven_multilingual_v2"})
                        return {"audio": base64.b64encode(r.content).decode(), "provider": "elevenlabs"}
                except: pass
            if OPENAI_API_KEY:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post("https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model": "tts-1", "voice": voice, "input": text})
                    return {"audio": base64.b64encode(r.content).decode(), "provider": "openai"}
            raise HTTPException(500, "No TTS provider")
        elif detected_intent == "translation":
            text_to_translate = prompt
            for pattern in [r"translate.*?to\s+(\w+):\s*(.+)", r"in\s+(\w+):\s*(.+)", r"to\s+(\w+):\s*(.+)"]:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match: target_language, text_to_translate = match.group(1), match.group(2); break
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": f"Translate to {target_language}:\n{text_to_translate}"}], "max_tokens": max_tokens, "temperature": 0.3})
                return {"original": text_to_translate, "translated": r.json()["choices"][0]["message"]["content"], "target_language": target_language}

        # DEFAULT CHAT STREAM
        user_memory_str = await fetch_user_memory(user_id)
        async def chat_gen():
            full_reply = ""
            try:
                yield sse({"type": "status", "status": "thinking"})
                messages = []
                base_system = system_prompt or "You are HeloXAI, an advanced AI assistant. Be helpful, accurate, and concise."
                if user_memory_str: base_system += f"\n\n{user_memory_str}\n\nUse this context to personalize if relevant."
                messages.append({"role": "system", "content": base_system})
                if context: messages.append({"role": "system", "content": f"Context: {context}"})
                if documents:
                    doc_texts = [await extract_document_text(d) for d in documents]
                    messages.append({"role": "system", "content": f"Documents:\n{chr(10).join(doc_texts)}"})
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})
                
                reply = await chat_with_tools(user_id, messages)
                full_reply = reply
                for char in reply:
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.008)
                
                await asyncio.to_thread(lambda: supabase.table("messages").insert({"id": str(uuid.uuid4()), "conversation_id": conversation_id, "user_id": user_id, "role": "assistant", "content": reply, "created_at": datetime.utcnow().isoformat()}).execute())
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Chat failed: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                if full_reply: asyncio.create_task(extract_and_save_memory(user_id, prompt, full_reply))
        return StreamingResponse(chat_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    except HTTPException: raise
    except Exception as e:
        logger.error(f"/ask/universal failed: {e}")
        raise HTTPException(500, str(e))


# ---------- TTS/STT ----------
@app.post("/tts")
async def text_to_speech(request: Request, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    text = re.sub(r"[#*`\[\]]", "", body.get("text", "")).strip()
    voice = body.get("voice", "alloy")
    if not text: raise HTTPException(400, "Text required")
    if ELEVENLABS_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"https://api.elevenlabs.io/v1/text-to-speech/{get_elevenlabs_voice_id(voice)}", headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}, json={"text": text, "model_id": "eleven_multilingual_v2"})
                return {"audio": base64.b64encode(r.content).decode(), "provider": "elevenlabs"}
        except: pass
    if OPENAI_API_KEY:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model": "tts-1", "voice": voice, "input": text})
            return {"audio": base64.b64encode(r.content).decode(), "provider": "openai"}
    raise HTTPException(500, "No TTS provider")

@app.post("/stt")
async def speech_to_text(request: Request, current_user: dict = Depends(get_current_user_optional)):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        audio_file = form.get("audio") or form.get("file")
        if not audio_file: raise HTTPException(400, "No audio file")
        audio_bytes = await audio_file.read()
    else:
        body = await request.json()
        audio_b64 = body.get("audio") or body.get("data")
        if not audio_b64: raise HTTPException(400, "No audio data")
        if "," in audio_b64: audio_b64 = audio_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(audio_b64)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    try:
        if OPENAI_API_KEY:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(temp_path, "rb") as f:
                    r = await client.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files={"file": ("audio.wav", f)}, data={"model": "whisper-1"})
                    return {"text": r.json()["text"], "provider": "openai"}
        raise HTTPException(500, "No STT provider")
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)

@app.get("/tts/voices")
async def get_tts_voices():
    return {"voices": {"openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], "elevenlabs": ["rachel", "drew", "bella", "antoni", "josh", "grace"]}, "providers": {"openai": bool(OPENAI_API_KEY), "elevenlabs": bool(ELEVENLABS_API_KEY)}}


# ---------- Session Management ----------
@app.post("/session/validate")
async def validate_session(current_user: dict = Depends(get_current_user_optional)):
    return {"valid": current_user.get("session_valid", False), "user_id": current_user.get("id")}

@app.post("/session/logout")
async def logout(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    user_id = current_user.get("id")
    if user_id:
        try:
            await asyncio.to_thread(lambda: supabase.table("user_sessions").update({"is_valid": False}).eq("user_id", user_id).execute())
        except: pass
        if user_id in _session_cache: del _session_cache[user_id]
    clear_session_cookies(response)
    return {"status": "logged_out"}


# ---------- Chat Management ----------
@app.post("/stop")
async def stop_generation(request: Request):
    body = await request.json()
    stream_id = body.get("stream_id")
    if stream_id:
        await stop_stream(stream_id)
        return {"status": "stopped"}
    async with active_stream_lock:
        for event in active_streams.values(): event.set()
        stopped = len(active_streams)
        active_streams.clear()
    return {"status": "stopped", "streams_stopped": stopped}

@app.post("/regenerate")
async def regenerate_response(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    conversation_id = body.get("conversation_id")
    if not conversation_id: raise HTTPException(400, "conversation_id required")
    user_id = (current_user or {}).get("id") or str(uuid.uuid4())
    
    messages_res = await asyncio.to_thread(lambda: supabase.table("messages").select("id, role, content").eq("conversation_id", conversation_id).eq("user_id", user_id).order("created_at", desc=True).limit(10).execute())
    messages = messages_res.data or []
    if not messages: raise HTTPException(404, "No messages")
    
    last_user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    last_assistant_id = next((m["id"] for m in messages if m["role"] == "assistant"), None)
    if not last_user_msg: raise HTTPException(400, "No user message")
    
    history = await asyncio.to_thread(lambda: supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).eq("user_id", user_id).order("created_at").limit(20).execute())
    history_messages = history.data or []
    if history_messages and history_messages[-1]["role"] == "assistant": history_messages = history_messages[:-1]
    
    if last_assistant_id:
        await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("id", last_assistant_id).execute())
    
    async def gen():
        try:
            yield sse({"type": "status", "status": "regenerating"})
            reply = await chat_with_tools(user_id, [{"role": "system", "content": "You are HeloXAI, a helpful AI assistant."}] + history_messages)
            for char in reply:
                yield sse({"type": "token", "text": char})
                await asyncio.sleep(0.008)
            await asyncio.to_thread(lambda: supabase.table("messages").insert({"id": str(uuid.uuid4()), "conversation_id": conversation_id, "user_id": user_id, "role": "assistant", "content": reply, "created_at": datetime.utcnow().isoformat()}).execute())
            yield sse({"type": "done"})
        except Exception as e: yield sse({"type": "error", "message": str(e)})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

@app.post("/newchat")
async def create_new_chat(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id") or str(uuid.uuid4())
    new_id = str(uuid.uuid4())
    await asyncio.to_thread(lambda: supabase.table("conversations").insert({"id": new_id, "user_id": user_id, "created_at": datetime.utcnow().isoformat()}).execute())
    return {"conversation_id": new_id, "user_id": user_id}

@app.delete("/chat/{conversation_id}")
async def delete_chat(conversation_id: str, request: Request, current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id")
    if not user_id: raise HTTPException(401, "Auth required")
    await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("conversation_id", conversation_id).eq("user_id", user_id).execute())
    await asyncio.to_thread(lambda: supabase.table("conversations").delete().eq("id", conversation_id).eq("user_id", user_id).execute())
    return {"status": "deleted"}

@app.get("/chats")
async def list_chats(request: Request, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id")
    if not user_id: return {"chats": [], "total": 0}
    conv_res = await asyncio.to_thread(lambda: supabase.table("conversations").select("id, title, created_at, updated_at").eq("user_id", user_id).order("updated_at", desc=True).range(offset, offset + limit - 1).execute())
    return {"chats": conv_res.data or [], "total": len(conv_res.data or [])}

@app.get("/chat/{conversation_id}")
async def get_chat(conversation_id: str, request: Request, limit: int = Query(50, ge=1, le=200), current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id")
    if not user_id: raise HTTPException(401, "Auth required")
    conv_res = await asyncio.to_thread(lambda: supabase.table("conversations").select("*").eq("id", conversation_id).eq("user_id", user_id).execute())
    if not conv_res.data: raise HTTPException(404, "Not found")
    msg_res = await asyncio.to_thread(lambda: supabase.table("messages").select("id, role, content, created_at").eq("conversation_id", conversation_id).order("created_at").range(0, limit - 1).execute())
    return {"conversation": conv_res.data[0], "messages": msg_res.data or []}

@app.patch("/chat/{conversation_id}")
async def update_chat(conversation_id: str, request: Request, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    user_id = (current_user or {}).get("id")
    if not user_id: raise HTTPException(401, "Auth required")
    update_data = {"updated_at": datetime.utcnow().isoformat()}
    if "title" in body: update_data["title"] = body["title"]
    if "system_prompt" in body: update_data["system_prompt"] = body["system_prompt"]
    await asyncio.to_thread(lambda: supabase.table("conversations").update(update_data).eq("id", conversation_id).eq("user_id", user_id).execute())
    return {"status": "updated"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/info")
async def info():
    return {"creator": CREATOR_INFO, "models": {"chat": CHAT_MODEL, "code": CODE_MODEL}, "providers": {"groq": bool(GROQ_API_KEY), "openai": bool(OPENAI_API_KEY), "elevenlabs": bool(ELEVENLABS_API_KEY), "judge0": bool(JUDGE0_KEY), "serper": bool(SERPER_API_KEY), "runway": bool(RUNWAYML_API_KEY)}, "features": ["streaming", "file_analysis", "archives", "code_execution", "web_search", "tts", "stt", "user_memory", "session_persistence"]}

@app.get("/capabilities")
async def capabilities():
    return {"intents": [i.value for i in IntentCategory], "file_types": {"code": len(CODE_EXTENSIONS), "document": len(DOCUMENT_EXTENSIONS), "archive": len(ARCHIVE_EXTENSIONS)}}

@app.get("/setup/sessions-table")
async def setup_sessions_table():
    return {"sql": """
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token TEXT NOT NULL,
    fingerprint TEXT,
    user_agent TEXT,
    ip_address TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(user_id, token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_valid ON user_sessions(user_id, is_valid) WHERE is_valid = TRUE;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own sessions" ON user_sessions FOR ALL USING (user_id = auth.uid());
"""}


if __name__ == "__main__":
    import uvicorn
    print(f"""
    ╔══════════════════════════════════════════╗
    ║      HeloXAI ULTIMATE Server v2.0         ║
    ╠══════════════════════════════════════════╣
    ║  Chat Model: {CHAT_MODEL:<30} ║
    ║  File Handling: Archives, Code, Docs     ║
    ║  Auth: Production Session Management      ║
    ║  API:        http://localhost:8000         ║
    ╚══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
