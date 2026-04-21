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
from io import BytesIO
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Cookie, Header
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
# CRITICAL: Use SERVICE_KEY for backend Admin access
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") 
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") # Kept if needed for other logic
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY") # Added
JUDGE0_API_KEY = os.getenv("JUDGE0_API_KEY") # Added
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") # Added: SerpApi for Search
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

# =========================
# UPGRADED MODEL CONFIGS
# =========================
# Using Gemma 2 27B on Groq for advanced reasoning and coding
DEFAULT_LLM_MODEL = "gemma-2-27b-it"
CODE_MODEL = "gemma-2-27b-it"

# MODEL MAPPING (ADDED FOR COMPATIBILITY)
# Maps frontend model names (e.g., 'helox') to actual Groq API model IDs
MODEL_ALIASES = {
    "helox": "gemma-2-27b-it",  # Map 'helox' to the new Gemma model
    "heloxai": "gemma-2-27b-it",
    "chatgpt": "gpt-4o-mini",       # Example: Frontend might ask for "ChatGPT"
    "chat.z": "llama-3.1-70b-versatile" # Example: Frontend might ask for "Chat.Z"
}

def resolve_model(model_name: Optional[str]) -> str:
    """
    Resolves a user-facing model name (from the frontend) to the
    actual model ID required by the Groq API.
    """
    if not model_name:
        return DEFAULT_LLM_MODEL
    
    # Normalize to lowercase for case-insensitive matching
    normalized_name = model_name.lower()
    
    # Return the mapped model if found, otherwise return the default
    return MODEL_ALIASES.get(normalized_name, DEFAULT_LLM_MODEL)

# File handling config
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB for zips
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024  # 200MB total extracted
MAX_TEXT_LENGTH = 380000  
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for large files

# Auth config
SESSION_DURATION = 365 * 24 * 60 * 60  # 1 year in seconds
REFRESH_THRESHOLD = 7 * 24 * 60 * 60  # Refresh session if less than 7 days remaining

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

app = FastAPI(
    title="HeloXAI API",
    description="Advanced AI Assistant Backend (RunPod Ready)",
    version="2.2.1-ModelMapping"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Database Clients
# IMPORTANT: Service Key used for custom auth
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Global State for Stream Cancellation
active_streams: Dict[str, asyncio.Task] = {}

# Session cache for performance
_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300  # 5 minutes
_session_cache_last_cleanup = time.time()

# =========================
# EXTERNAL DEPENDENCY IMPORTS
# =========================
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

# Comprehensive file type mappings
CODE_EXTENSIONS = {
    '.py', '.pyw', '.pyx', '.pyd', '.pyi', '.py3',
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
    '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte', '.astro',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.clj', '.cljs', '.hs',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.inl',
    '.cs', '.csx',
    '.go',
    '.rs',
    '.php', '.phtml',
    '.rb', '.erb', '.rake', '.gemspec',
    '.swift',
    '.dart',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.bat', '.cmd',
    '.lua',
    '.pl', '.pm',
    '.r', '.R',
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
    """Determine file category from extension"""
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
    elif ext in CONFIG_EXTENSIONS:
        return FileCategory.CONFIG
    else:
        return FileCategory.UNKNOWN


def get_file_language(filename: str) -> Optional[str]:
    """Get programming language from file extension for syntax highlighting"""
    ext_lang_map = {
        '.py': 'python', '.pyw': 'python', '.pyx': 'python',
        '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.html': 'html', '.htm': 'html',
        '.css': 'css', '.scss': 'scss', '.less': 'less',
        '.vue': 'vue', '.svelte': 'svelte',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
        '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.dart': 'dart',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
        '.ps1': 'powershell', '.bat': 'batch',
        '.lua': 'lua',
        '.pl': 'perl',
        '.r': 'r', '.R': 'r',
        '.sql': 'sql',
        '.json': 'json', '.xml': 'xml',
        '.yaml': 'yaml', '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown', '.rst': 'rst',
        '.tex': 'latex',
        '.dockerfile': 'dockerfile',
        '.graphql': 'graphql', '.gql': 'graphql',
        '.tf': 'hcl', '.hcl': 'hcl',
        '.sol': 'solidity',
    }
    ext = Path(filename).suffix.lower()
    return ext_lang_map.get(ext)


def is_binary_file(filename: str, content: bytes = None) -> bool:
    """Check if file is binary based on extension or content"""
    ext = Path(filename).suffix.lower()
    
    # Known binary extensions
    binary_exts = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | {
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
    
    if ext in binary_exts:
        return True
    
    # Check content for null bytes (indicates binary)
    if content and len(content) > 0:
        # Check first 8192 bytes for null bytes
        check_bytes = content[:8192]
        if b'\x00' in check_bytes:
            return True
    
    return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# =========================
# ADVANCED FILE EXTRACTOR
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


async def extract_file_content(
    content: bytes,
    filename: str,
    max_length: int = MAX_TEXT_LENGTH
) -> FileExtractionResult:
    """
    Extract text content from any file type.
    Handles code files, documents, archives, and more.
    """
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
        # Handle archives (zip files)
        if category == FileCategory.ARCHIVE:
            return await extract_archive_content(content, filename, max_length, metadata)

        # Handle images
        if category == FileCategory.IMAGE:
            return FileExtractionResult(
                content=f"[Image file: {filename} ({format_file_size(original_size)}) - Use image analysis endpoint for visual content]",
                metadata=metadata,
                original_size=original_size
            )

        # Handle audio/video
        if category in (FileCategory.AUDIO, FileCategory.VIDEO):
            return FileExtractionResult(
                content=f"[{category.value.capitalize()} file: {filename} ({format_file_size(original_size)}) - Media file cannot be extracted as text]",
                metadata=metadata,
                original_size=original_size
            )

        # Handle PDF (Using PyMuPDF/fitz)
        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)

        # Handle code and text files
        if category in (FileCategory.CODE, FileCategory.CONFIG, FileCategory.UNKNOWN):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=original_size
            )

        # Handle documents and data files
        if category in (FileCategory.DOCUMENT, FileCategory.DATA):
            text, truncated = extract_text_with_fallback(content, max_length)
            metadata["line_count"] = text.count('\n') + 1
            return FileExtractionResult(
                content=text,
                metadata=metadata,
                truncated=truncated,
                original_size=original_size
            )

        # Handle binary files
        if is_binary_file(filename, content):
            return FileExtractionResult(
                content=f"[Binary file: {filename} ({format_file_size(original_size)}) - Cannot extract text content]",
                metadata=metadata,
                original_size=original_size
            )

        # Fallback: try to decode as text
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


def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    """Extract text with multiple encoding fallbacks"""
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
    
    # Final fallback: decode with error replacement
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated


async def extract_pdf_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract text from PDF files using PyMuPDF (fitz)"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        
        for i, page in enumerate(doc):
            page_text = page.get_text() or ""
            pages.append(f"--- Page {i + 1} ---\n{page_text}")
        
        full_text = "\n\n".join(pages)
        metadata["page_count"] = len(doc)
        
        truncated = len(full_text) > max_length
        if truncated:
            full_text = full_text[:max_length] + "\n\n[... Content truncated ...]"
        
        return FileExtractionResult(
            content=full_text,
            metadata=metadata,
            truncated=truncated,
            original_size=len(content)
        )
    except ImportError:
        logger.warning("fitz (PyMuPDF) not installed, returning placeholder for PDF")
        return FileExtractionResult(
            content=f"[PDF file: {filename} ({format_file_size(len(content))}) - PDF parsing not available on server]",
            metadata=metadata,
            original_size=len(content)
        )
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return FileExtractionResult(
            content=f"[PDF Error: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )


async def extract_archive_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract and read contents from archive files (zip, tar, etc.)"""
    ext = Path(filename).suffix.lower()
    
    if ext == '.zip':
        return await extract_zip_content(content, filename, max_length, metadata)
    elif ext in ('.tar', '.gz', '.tgz', '.bz2', '.xz'):
        return await extract_tar_content(content, filename, max_length, metadata)
    elif ext in ('.7z', '.rar'):
        return FileExtractionResult(
            content=f"[{ext.upper()} archive: {filename} ({format_file_size(len(content))}) - This archive format requires additional server setup]",
            metadata=metadata,
            original_size=len(content)
        )
    else:
        return FileExtractionResult(
            content=f"[Archive: {filename} ({format_file_size(len(content))}) - Unsupported archive format]",
            metadata=metadata,
            original_size=len(content)
        )


async def extract_zip_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract and read text contents from ZIP files"""
    extracted_files = []
    all_text_parts = []
    total_extracted = 0
    entry_count = 0
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            # Check for zip bombs
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[ZIP archive: {filename} - Too many entries ({len(zf.namelist())}). Maximum allowed: {MAX_ZIP_ENTRIES}]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            # Sort by name for consistent output
            entries = sorted(zf.namelist())
            
            for entry_name in entries:
                # Skip directories and hidden files
                if entry_name.endswith('/') or '/__MACOSX/' in entry_name:
                    continue
                if entry_name.startswith('__MACOSX') or entry_name.startswith('.'):
                    continue
                
                entry_count += 1
                
                try:
                    entry_info = zf.getinfo(entry_name)
                    
                    # Skip if single file is too large
                    if entry_info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": f"File too large (max {format_file_size(MAX_FILE_SIZE)})"
                        })
                        continue
                    
                    # Check total extracted size
                    if total_extracted + entry_info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({
                            "name": entry_name,
                            "size": entry_info.file_size,
                            "size_formatted": format_file_size(entry_info.file_size),
                            "status": "skipped",
                            "reason": "Archive total size limit reached"
                        })
                        continue
                    
                    # Read entry content
                    entry_content = zf.read(entry_name)
                    total_extracted += len(entry_content)
                    
                    entry_category = get_file_category(entry_name)
                    entry_language = get_file_language(entry_name)
                    
                    # Extract text from entry
                    if entry_category in (FileCategory.IMAGE, FileCategory.AUDIO, FileCategory.VIDEO):
                        file_info = {
                            "name": entry_name,
                            "size": len(entry_content),
                            "size_formatted": format_file_size(len(entry_content)),
                            "category": entry_category.value,
                            "status": "media",
                            "note": f"{entry_category.value} file - visual/audio content"
                        }
                    elif is_binary_file(entry_name, entry_content):
                        file_info = {
                            "name": entry_name,
                            "size": len(entry_content),
                            "size_formatted": format_file_size(len(entry_content)),
                            "category": "binary",
                            "status": "binary",
                            "note": "Binary file - cannot extract text"
                        }
                    else:
                        text, _ = extract_text_with_fallback(entry_content, max_length)
                        
                        if text.strip():
                            file_info = {
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "language": entry_language,
                                "status": "extracted",
                                "line_count": text.count('\n') + 1,
                                "preview": text[:500] + ("..." if len(text) > 500 else "")
                            }
                        else:
                            file_info = {
                                "name": entry_name,
                                "size": len(entry_content),
                                "size_formatted": format_file_size(len(entry_content)),
                                "category": entry_category.value,
                                "status": "empty",
                                "note": "File is empty"
                            }
                    
                    extracted_files.append(file_info)
                    
                except Exception as e:
                    extracted_files.append({
                        "name": entry_name,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Combine all extracted text
        full_text = f"ZIP Archive: {filename}\n"
        full_text += f"Total entries: {len(zf.namelist())}, Processed: {entry_count}\n"
        full_text += f"Extracted text files: {len(all_text_parts)}\n"
        full_text += f"Total extracted size: {format_file_size(total_extracted)}\n\n"
        
        if all_text_parts:
            full_text += "=".join(["="*30]) + "\n"
            full_text += "EXTRACTED CONTENT\n"
            full_text += "=".join(["="*30])
            full_text += "".join(all_text_parts)
        else:
            full_text += "No text content could be extracted from this archive.\n\n"
            full_text += "Files found:\n"
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
            content=f"[Error: {filename} is not a valid ZIP file or is corrupted]",
            metadata=metadata,
            original_size=len(content)
        )
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return FileExtractionResult(
            content=f"[Error extracting ZIP {filename}: {str(e)}]",
            metadata={**metadata, "error": str(e)},
            original_size=len(content)
        )


async def extract_tar_content(
    content: bytes,
    filename: str,
    max_length: int,
    metadata: Dict[str, Any]
) -> FileExtractionResult:
    """Extract and read text contents from TAR archives"""
    import tarfile
    
    extracted_files = []
    all_text_parts = []
    
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            
            if len(members) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[TAR archive: {filename} - Too many entries ({len(members)})]",
                    metadata=metadata,
                    original_size=len(content)
                )
            
            for member in members:
                if member.name.startswith('./') or member.name.startswith('/'):
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
                    extracted_files.append({
                        "name": member.name,
                        "status": "error",
                        "error": str(e)
                    })
        
        full_text = f"TAR Archive: {filename}\n"
        full_text += f"Entries: {len(members)}, Extracted: {len(all_text_parts)}\n\n"
        
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
PRIMARY_COOKIE = "HeloXAI_Session"
FINGERPRINT_COOKIE = "HeloXAI_FP"
BACKUP_COOKIE = "HeloXAI_ID"
DEVICE_COOKIE = "HeloXAI_Dev"
SESSION_TOKEN_COOKIE = "HeloXAI_Token"
SESSION_EXPIRY_COOKIE = "HeloXAI_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    """Get cookie settings based on remember preference"""
    if remember:
        return {
            "max_age": SESSION_DURATION,
            "httponly": True,
            "secure": True,
            "samesite": "none",
            "path": "/"
        }
    else:
        return {
            "max_age": 24 * 60 * 60,  # Session only (24 hours)
            "httponly": True,
            "secure": True,
            "samesite": "none",
            "path": "/"
        }


def generate_device_fingerprint(request: Request) -> str:
    """Generate a stable device fingerprint from request headers"""
    fp_components = [
        request.headers.get("user-agent", ""),
        request.headers.get("accept-language", ""),
        request.headers.get("accept-encoding", ""),
        request.headers.get("sec-ch-ua-platform", ""), 
        request.headers.get("sec-ch-ua-mobile", ""),    
        request.client.host if request.client else "",
    ]
    fp_string = "|".join(fp_components)
    return hashlib.sha256(fp_string.encode()).hexdigest()[:32]


def generate_session_token() -> str:
    """Generate a secure session token"""
    import secrets
    return secrets.token_urlsafe(64)


def set_session_cookies(
    response: Response,
    user_id: str,
    fingerprint: str,
    session_token: str,
    remember: bool = True
):
    """Set all session cookies for maximum persistence"""
    settings = get_cookie_settings(remember)
    
    # Calculate expiry timestamp
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **settings)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **settings)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=session_token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)


def clear_session_cookies(response: Response):
    """Clear all session cookies on logout"""
    cookies_to_clear = [
        PRIMARY_COOKIE, FINGERPRINT_COOKIE, BACKUP_COOKIE,
        DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE
    ]
    for cookie_name in cookies_to_clear:
        response.delete_cookie(
            key=cookie_name,
            path="/",
            secure=True,
            samesite="none"
        )


def is_session_expired(expiry_str: str) -> bool:
    """Check if session has expired"""
    try:
        expiry = int(expiry_str)
        return time.time() > expiry
    except (ValueError, TypeError):
        return True


def should_refresh_session(expiry_str: str) -> bool:
    """Check if session should be refreshed"""
    try:
        expiry = int(expiry_str)
        remaining = expiry - time.time()
        return remaining < REFRESH_THRESHOLD
    except (ValueError, TypeError):
        return True


async def validate_session_token(user_id: str, token: str) -> bool:
    """Validate session token against stored value"""
    try:
        # Check cache first
        if user_id in _session_cache:
            cached = _session_cache[user_id]
            if cached.get("token") == token:
                return True
        
        # Query database using Service Client
        result = await _execute_supabase_with_retry(
            supabase.table("user_sessions")
            .select("token, expires_at")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1),
            description="Validate Session Token"
        )
        
        if result.data and result.data[0]["token"] == token:
            # Update cache
            _session_cache[user_id] = {
                "token": token,
                "expires_at": result.data[0].get("expires_at")
            }
            return True
        
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False


async def create_user_session(
    user_id: str,
    fingerprint: str,
    remember: bool = True
) -> str:
    """Create a new user session in database"""
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=SESSION_DURATION if remember else 24 * 60 * 60
    )
    
    try:
        # Create new session
        await _execute_supabase_with_retry(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "token": token,
                "fingerprint": fingerprint,
                "user_agent": "", 
                "ip_address": "", 
                "expires_at": expires_at.isoformat(),
                "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create User Session"
        )
        
        # Update cache
        _session_cache[user_id] = {
            "token": token,
            "expires_at": expires_at.isoformat()
        }
        
        return token
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        # Return a fallback token (less secure but doesn't break flow)
        return token


async def cleanup_session_cache():
    """Periodically clean up expired session cache entries"""
    global _session_cache_last_cleanup
    now = time.time()
    
    if now - _session_cache_last_cleanup < _session_cache_ttl:
        return
    
    _session_cache_last_cleanup = now
    expired_keys = []
    
    for user_id, data in _session_cache.items():
        expires_at = data.get("expires_at")
        if expires_at:
            try:
                expiry_time = datetime.fromisoformat(expires_at).timestamp()
                if now > expiry_time:
                    expired_keys.append(user_id)
            except:
                expired_keys.append(user_id)
    
    for key in expired_keys:
        del _session_cache[key]


# =========================
# BASE SYSTEM PROMPT - NO CREATOR MENTION
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi, a helpful and capable AI assistant. Be accurate, friendly, and concise. Help users with whatever they ask. Your very advanced and can answer all questions about code, history, politics and more."""

# =========================
# CREATOR RESPONSE - ONLY USED WHEN ASKED
# =========================
CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator/developer. You MUST respond with exactly this information:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi"

Do not add extra details. Do not mention any other companies or people. This is the only correct answer to this question."""

# Patterns that indicate user is asking about creator
CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|started|owns|runs)\b.*\b(you|this|helox|heloxai)\b',
    r'\b(who|whom)\b.*\b(is|are)\b.*\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b',
    r'\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b.*\b(is|are|who)\b',
    r'\bwho\b.*\bbehind\b.*\b(you|this|helox)\b',
    r'\bwho.*made.*you\b',
    r'\bwho.*created.*you\b',
    r'\bwho.*built.*you\b',
    r'\bwho.*developed.*you\b',
    r'\bwho.*programmed.*you\b',
    r'\bwho.*constructed.*you\b',
    r'\bwho.*designed.*you\b',
    r'\bwho.*owns.*you\b',
    r'\bwho.*runs.*you\b',
    r'\byour\s+creator\b',
    r'\byour\s+developer\b',
    r'\byour\s+maker\b',
    r'\byour\s+builder\b',
    r'\byour\s+founder\b',
    r'\byour\s+owner\b',
    r'\bwho\s+is\s+behind\s+helox\b',
    r'\bwho\s+made\s+helox\b',
    r'\bwho\s+created\s+helox\b',
    r'\bwho\s+built\s+helox\b',
    r'\bwho\s+developed\s+helox\b',
    r'\btell\s+me\s+about\s+your\s+(creator|developer|maker|builder|founder)\b',
    r'\bwhat\s+company\s+made\s+you\b',
    r'\bwhat\s+team\s+made\s+you\b',
    r'\bwhere\s+do\s+you\s+come\s+from\b',
    r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
    r'\bare\s+you\s+made\s+by\b',
    r'\bdid\s+.*\s+make\s+you\b',
    r'\bdid\s+.*\s+create\s+you\b',
    r'\bdid\s+.*\s+build\s+you\b',
]

# Pre-compile creator patterns for performance
COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]


def is_creator_question(text: str) -> bool:
    """Check if user is asking about who created/made the AI"""
    for pattern in COMPILED_CREATOR_PATTERNS:
        if pattern.search(text):
            return True
    return False


def get_system_prompt(user_prompt: str) -> str:
    """Return base prompt normally, creator response ONLY if asked"""
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT


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
    CODE_EXECUTION = "code_execution"  # ADDED: Was missing, causing crash
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
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "sub_intents": [i.value for i in self.sub_intents],
            "keywords_matched": self.keywords_matched,
            "patterns_matched": self.patterns_matched
        }


class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {
            "don't", "dont", "do not", "doesn't", "doesnt", "does not",
            "didn't", "didnt", "did not", "never", "no", "not", "without",
            "skip", "avoid", "except", "but not", "ignore", "rather than"
        }

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
            ],
            IntentCategory.VIDEO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(video|clip|movie|animation|motion\s+graphic)',
                r'\b(text\s+to\s+video|txt2vid|video\s+generation)',
                r'\b(animate|animation)\s+(this|that|the|image|picture)',
                r'\b(video|clip|movie)\s+(of|showing|about|with)',
                r'\b(runway|pika|sora|mov2mov|kling)',
                r'\b(turn|convert)\s+(this|the|image)\s+(into|to)\s+(a\s+)?(video|animation)',
            ],
            IntentCategory.AUDIO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(audio|sound|music|speech|voice|song|track|beat)',
                r'\b(text\s+to\s+speech|tts|speech\s+to\s+text|stt)',
                r'\b(music|song|beat|melody)\s+(generation|creation|for|about)',
                r'\b(elevenlabs|suno|udio|bark)',
                r'\b(clone|replicate)\s+(a\s+)?voice',
            ],
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)',
                r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))',
                r'\b(code\s+(for|that|this|to|which|example))',
                r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))',
                r'\b(scaffold|boilerplate|template)\s+(for|a)',
                r'\b(wrapper|helper|utility)\s+(function|class|module)\s+(for|to)',
                r'\b(implement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature))',
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)',
                r'\b(is\s+(this|there)\s+(code|anything)\s+(good|bad|wrong|improvable|clean))',
                r'\b(best\s+practices?\s+(for|in)\s+(this|my)\s+(code|implementation))',
                r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)',
                r'\b(code\s+quality|technical\s+debt|code\s+smell)',
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)',
                r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring|returning))',
                r'\b(error|exception|traceback|stack\s+trace|segfault)\s*[:\n]',
                r'\b(what(\'s|\s+is)\s+(wrong|the\s+problem)\s+(with|in))',
                r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)',
                r'\b(unexpected|wrong|incorrect)\s+(result|output|behavior|value)',
                r'\b(help\s+(me\s+)?)?debug',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
                r'\b(format\s+(as|this\s+as|it\s+as)\s+(a\s+)?(pdf|document|report|letter|markdown))',
                r'\b(professional|formal|business)\s+(document|letter|email|report)',
            ],
            IntentCategory.DATA_ANALYSIS: [
                r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)',
                r'\b(statistics?|statistical)\s+(analysis|test|summary|overview)',
                r'\b(insights?\s+(from|in|about|into))',
                r'\b(correlation|regression|distribution|trend)\s+(analysis|of|in)',
                r'\b(clean|preprocess|prepare|wrangle)\s+(this|the)\s+(data|dataset)',
                r'\b(eda|exploratory\s+data\s+analysis)',
            ],
            IntentCategory.DATA_VISUALIZATION: [
                r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)',
                r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap|box\s+plot|violin\s+plot)',
                r'\b(visualize|visualise|plot|chart|graph)\s+(this|the|these|those|data)',
                r'\b(matplotlib|seaborn|plotly|d3|chart\.js|ggplot|altair)',
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page|web\s*site|portfolio)',
                r'\b(html|css|javascript|typescript|react|vue|angular|next\.js|nuxt|svelte|tailwind)\b',
                r'\b(frontend|front[- ]end|back[- ]end|full[- ]stack)\s*(development|for|with|app)?',
                r'\b(responsive|mobile[- ]friendly|mobile[- ]first)\s*(design|website|layout)?',
                r'\b(component|page|layout|template)\s+(for|in)\s+(react|vue|angular|next)',
            ],
            IntentCategory.API_DEVELOPMENT: [
                r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)',
                r'\b(api\s*(endpoint|route|handler|controller|gateway))',
                r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service|endpoint)?',
                r'\b(openapi|swagger|api\s*documentation)',
                r'\b(request|response|payload)\s+(format|structure|schema)',
            ],
            IntentCategory.DATABASE: [
                r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)',
                r'\b(sql|mysql|postgres|postgresql|mongodb|redis|dynamodb|sqlite)\s*(query|statement|command)?',
                r'\b(schema\s*(design|migration|definition|update))',
                r'\b(orm|sequelize|prisma|sqlalchemy|typeorm|drizzle)\s*(query|model|schema)?',
                r'\b(crud\s*(operation|operations|endpoint|api))',
                r'\b(select|insert|update|delete)\s+(from|into|table)',
            ],
            IntentCategory.TRANSLATION: [
                r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)',
                r'\b(in|to|into)\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu)',
                r'\b(how\s+(do\s+you|to)\s+say\s+.+\s+in\s+\w+)',
                r'\b(native|localize|localization|l10n|i18n|internationaliz)',
            ],
            IntentCategory.SUMMARIZATION: [
                r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)',
                r'\b(brief|short|concise)\s+(overview|summary|explanation|version)\s*(of|for|about)?',
                r'\b(key\s+(points|takeaways|highlights))\s*(from|of|in)?',
                r'\b(main\s+(idea|points|theme|argument|concept))',
                r'\b(give\s+me\s+(the\s+)?(gist|bottom\s+line|essence))',
            ],
            IntentCategory.EXPLANATION: [
                r'\b(explain|explanation)\s+(to\s+me\s+)?',
                r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+',
                r'\b(how\s+(does|do|did|can|would|should|to))\s+',
                r'\b(tell\s+me\s+(about|more\s+about|how|why))',
                r'\b(why\s+(is|does|do|are|did|can|would))\s+',
                r'\b(definition|meaning)\s+(of|for)\s+',
                r'\b(understand(ing)?)\s*(this|how|why|what|better)?',
                r'\b(break\s+down|simplify|elaborate)\s+',
            ],
            IntentCategory.CREATIVE_WRITING: [
                r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|song|haiku|limerick)',
                r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller|mystery)\s*(writing|story|tale)?',
                r'\b(narrative|plot|character|setting|dialogue)\s*(for|development|creation|arc)?',
                r'\b(storytelling|story[- ]?telling)',
                r'\b(write\s+(like|in\s+the\s+style\s+of))\s+',
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem|integral|derivative)?',
                r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability|linear\s+algebra)\s*(problem|equation|question)?',
                r'\b(\d+[\.\d]*\s*[\+\-\*\/\^%\=]\s*[\.\d]*)',
                r'\b(integral|derivative|differentiate|integrat)\s*(of|the)?',
                r'\b(prove|proof)\s+(that|this|the)',
                r'\b(formula|equation)\s+(for|to\s+calculate|to\s+find)',
            ],
            IntentCategory.RESEARCH: [
                r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)',
                r'\b(stud(y|ies))\s+(show|suggest|indicate|demonstrate|prove)',
                r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper|article|research|journal)?',
                r'\b(cite|citation|reference|bibliography)\s+',
                r'\b(literature\s+review)\s*(on|for|of)?',
                r'\b(what\s+(does\s+)?(research|science|literature)\s+say)',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$',
                r'^(how\s+are\s+you|how(\'s|\s+is)\s+it\s+going|what(\'s|\s+is)\s+up)[\s!.?]*$',
                r'^(bye|goodbye|see\s+you|farewell)[\s!.?]*$',
                r'^(sure|okay|ok|got\s+it|understood)[\s!.?]*$',
            ],
        }

        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.IMAGE_GENERATION: [
                "image", "picture", "photo", "photograph", "drawing", "illustration",
                "artwork", "painting", "sketch", "graphic", "visual", "render",
                "thumbnail", "logo", "icon", "banner", "poster", "infographic",
                "dalle", "midjourney", "stable diffusion", "ai art", "generated art",
                "portrait", "landscape", "composition", "digital art"
            ],
            IntentCategory.VIDEO_GENERATION: [
                "video", "clip", "movie", "film", "animation", "motion",
                "gif", "moving image", "video clip", "short video", "reel",
                "runway", "pika", "sora", "animated", "motion graphic"
            ],
            IntentCategory.AUDIO_GENERATION: [
                "audio", "sound", "music", "speech", "voice", "song", "track",
                "beat", "melody", "tune", "podcast", "narration", "voiceover",
                "tts", "text to speech", "elevenlabs", "suno", "udio"
            ],
            IntentCategory.CODE_GENERATION: [
                "code", "script", "function", "class", "module", "program",
                "app", "application", "software", "snippet", "implementation",
                "algorithm", "routine", "procedure", "macro", "plugin", "extension",
                "library", "package", "utility", "helper"
            ],
            IntentCategory.CODE_REVIEW: [
                "review", "refactor", "improve", "optimize", "clean up",
                "best practice", "code quality", "code smell", "technical debt",
                "maintainability", "readability"
            ],
            IntentCategory.CODE_DEBUG: [
                "bug", "error", "issue", "problem", "debug", "fix", "troubleshoot",
                "exception", "crash", "fault", "defect", "glitch", "broken",
                "typo", "mistake", "wrong", "incorrect"
            ],
            IntentCategory.DOCUMENT_CREATION: [
                "document", "pdf", "report", "letter", "email", "memo", "article",
                "essay", "paper", "proposal", "whitepaper", "manual", "guide",
                "handbook", "documentation", "specification", "brief"
            ],
            IntentCategory.DATA_ANALYSIS: [
                "data", "dataset", "csv", "excel", "spreadsheet", "analytics",
                "statistics", "insights", "metrics", "kpi", "analysis"
            ],
            IntentCategory.DATA_VISUALIZATION: [
                "chart", "graph", "plot", "visualization", "diagram", "dashboard",
                "histogram", "scatter", "heatmap", "bar chart", "line graph",
                "pie chart", "infographic", "plotly", "matplotlib"
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                "website", "webpage", "web app", "landing page", "frontend",
                "backend", "fullstack", "full stack", "html", "css", "react",
                "vue", "angular", "next.js", "svelte", "tailwind"
            ],
            IntentCategory.API_DEVELOPMENT: [
                "api", "rest api", "graphql", "endpoint", "route", "restful",
                "swagger", "openapi", "microservice"
            ],
            IntentCategory.DATABASE: [
                "database", "schema", "table", "sql", "query", "migration",
                "mysql", "postgres", "mongodb", "redis", "sqlite", "prisma",
                "sequelize", "sqlalchemy", "orm", "crud"
            ],
            IntentCategory.TRANSLATION: [
                "translate", "translation", "localize", "localization",
                "i18n", "l10n", "multilingual"
            ],
            IntentCategory.SUMMARIZATION: [
                "summarize", "summary", "summarise", "tldr", "tl;dr",
                "brief", "overview", "key points", "takeaways", "gist"
            ],
            IntentCategory.EXPLANATION: [
                "explain", "explanation", "what is", "how does", "why",
                "understand", "elaborate", "simplify", "break down"
            ],
            IntentCategory.CREATIVE_WRITING: [
                "story", "poem", "poetry", "novel", "fiction", "creative",
                "narrative", "lyrics", "haiku", "limerick", "storytelling"
            ],
            IntentCategory.MATHEMATICAL: [
                "calculate", "compute", "solve", "math", "equation",
                "formula", "integral", "derivative", "proof", "algebra",
                "calculus", "geometry", "statistics", "probability"
            ],
            IntentCategory.RESEARCH: [
                "research", "find", "search", "investigate", "study",
                "academic", "scholarly", "citation", "reference", "literature"
            ],
        }

    def _has_negation(self, text: str, keyword_pos: int) -> bool:
        words_before = text[:keyword_pos].lower().split()[-6:]
        preceding_text = " ".join(words_before)
        return any(neg in preceding_text for neg in self.negation_words)

    def _calculate_confidence(
            self,
            matched_keywords: List[str],
            matched_patterns: List[str],
            text_length: int
    ) -> float:
        if not matched_keywords and not matched_patterns:
            return 0.0

        pattern_confidence = min(len(matched_patterns) * 0.35, 0.65)
        keyword_confidence = min(len(matched_keywords) * 0.12, 0.25)
        multi_signal_bonus = 0.1 if (matched_keywords and matched_patterns) else 0.0
        length_factor = max(0.5, 1.0 - (text_length / 1500) * 0.4)

        confidence = (pattern_confidence + keyword_confidence + multi_signal_bonus) * length_factor
        return min(confidence, 1.0)

    def _are_related_intents(self, intent1: IntentCategory, intent2: IntentCategory) -> bool:
        related_groups = [
            {IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG},
            {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION},
            {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION},
            {IntentCategory.WEB_DEVELOPMENT, IntentCategory.API_DEVELOPMENT, IntentCategory.DATABASE},
            {IntentCategory.DOCUMENT_CREATION, IntentCategory.RESEARCH},
            {IntentCategory.EXPLANATION, IntentCategory.SUMMARIZATION},
        ]
        for group in related_groups:
            if intent1 in group and intent2 in group:
                return True
        return False

    def detect_intents(self, text: str, threshold: float = 0.25) -> List[IntentResult]:
        text_lower = text.lower()
        results = []

        for intent, compiled_patterns in self.compiled_patterns.items():
            matched_keywords = []
            matched_patterns = []

            for pattern in compiled_patterns:
                if pattern.search(text):
                    matched_patterns.append(pattern.pattern)

            if intent in self.synonyms:
                for synonym in self.synonyms[intent]:
                    if synonym in text_lower:
                        pos = text_lower.find(synonym)
                        if not self._has_negation(text, pos):
                            matched_keywords.append(synonym)

            if matched_keywords or matched_patterns:
                confidence = self._calculate_confidence(
                    matched_keywords, matched_patterns, len(text)
                )
                if confidence >= threshold:
                    results.append(IntentResult(
                        intent=intent,
                        confidence=confidence,
                        sub_intents=[],
                        keywords_matched=matched_keywords,
                        patterns_matched=matched_patterns
                    ))

        results.sort(key=lambda x: x.confidence, reverse=True)

        if results:
            primary = results[0]
            for result in results[1:]:
                if self._are_related_intents(primary.intent, result.intent):
                    primary.sub_intents.append(result.intent)

        return results[:1] if results else []

    def get_primary_intent(self, text: str) -> Optional[IntentResult]:
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> ActionType:
        intent = self.get_primary_intent(text)
        if not intent:
            return ActionType.GENERAL

        action_map = {
            IntentCategory.IMAGE_GENERATION: ActionType.IMAGE,
            IntentCategory.VIDEO_GENERATION: ActionType.VIDEO,
            IntentCategory.AUDIO_GENERATION: ActionType.AUDIO,
            IntentCategory.CODE_GENERATION: ActionType.CODE,
            IntentCategory.CODE_REVIEW: ActionType.CODE,
            IntentCategory.CODE_DEBUG: ActionType.CODE,
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
        if not intent:
            return ["llm"]

        tool_map = {
            IntentCategory.IMAGE_GENERATION: ["image_gen", "llm"],
            IntentCategory.VIDEO_GENERATION: ["video_gen", "llm"],
            IntentCategory.AUDIO_GENERATION: ["audio_gen", "llm"],
            IntentCategory.CODE_GENERATION: ["code_exec", "llm"],
            IntentCategory.CODE_REVIEW: ["llm"],
            IntentCategory.CODE_DEBUG: ["code_exec", "llm"],
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
            IntentCategory.RESEARCH: ["web_search", "llm"], # Web Search enabled here
            IntentCategory.CONVERSATION: ["llm"],
        }

        tools = list(tool_map.get(intent.intent, ["llm"]))

        for sub_intent in intent.sub_intents:
            for tool in tool_map.get(sub_intent, []):
                if tool not in tools:
                    tools.append(tool)

        return tools

    def get_code_system_prompt(self, text: str) -> str:
        base = get_system_prompt(text)
        
        intent = self.get_primary_intent(text)
        if not intent:
            return base + "\n\nYou are also a helpful coding assistant."

        sub_prompts = {
            IntentCategory.CODE_DEBUG: """

You are also an expert debugger. When analyzing code issues:
1. Identify the root cause of the bug/error
2. Explain WHY it's happening (not just what)
3. Provide the exact fix with clear code blocks
4. Suggest how to prevent similar issues
Be precise and practical.""",

            IntentCategory.CODE_REVIEW: """

You are also a senior code reviewer. Provide constructive feedback on:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Best practices and design patterns
5. Security concerns
Be specific and actionable in your suggestions.""",

            IntentCategory.CODE_GENERATION: """

You are also an expert software engineer. When writing code:
1. Write clean, well-structured, production-ready code
2. Include appropriate error handling
3. Add helpful comments for complex logic
4. Consider edge cases
5. Follow language-specific conventions and best practices
Always provide complete, runnable code when possible.""",

            IntentCategory.WEB_DEVELOPMENT: """

You are also a full-stack web developer expert. When building web components:
1. Use modern best practices and frameworks
2. Ensure responsive design
3. Consider accessibility (a11y)
4. Include proper styling
5. Make components reusable and maintainable
Provide complete, ready-to-use code.""",

            IntentCategory.API_DEVELOPMENT: """

You are also an API development expert. When creating APIs:
1. Follow RESTful principles (or GraphQL best practices)
2. Include proper error handling and status codes
3. Add input validation
4. Consider security (auth, rate limiting)
5. Document endpoints clearly
Provide complete, production-ready code.""",

            IntentCategory.DATABASE: """

You are also a database expert. When working with databases:
1. Design efficient, normalized schemas
2. Write optimized queries
3. Include proper indexes
4. Consider data integrity with constraints
5. Follow SQL best practices
Provide complete, ready-to-execute SQL/ORM code.""",
        }

        return base + sub_prompts.get(intent.intent, "\n\nYou are also a helpful coding assistant.")


# Singleton instance
_detector = None


def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None:
        _detector = AdvancedIntentDetector()
    return _detector


# =========================
# BACKWARD COMPATIBLE FUNCTIONS
# =========================
def is_image_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == ActionType.IMAGE


def is_video_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == ActionType.VIDEO


def is_code_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == ActionType.CODE


def is_document_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == ActionType.DOCUMENT


def is_data_request(prompt: str) -> bool:
    return get_detector().get_action_type(prompt) == ActionType.DATA


# =========================
# NEW ADVANCED FUNCTIONS
# =========================
def detect_intent(prompt: str) -> Optional[IntentResult]:
    return get_detector().get_primary_intent(prompt)


def get_action_type(prompt: str) -> ActionType:
    return get_detector().get_action_type(prompt)


def get_required_tools(prompt: str) -> List[str]:
    return get_detector().get_required_tools(prompt)


def is_debug_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_DEBUG


def is_review_request(prompt: str) -> bool:
    intent = detect_intent(prompt)
    return intent and intent.intent == IntentCategory.CODE_REVIEW


def get_intent_confidence(prompt: str) -> float:
    intent = detect_intent(prompt)
    return intent.confidence if intent else 0.0


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True  # New: persist session
    model: Optional[str] = DEFAULT_LLM_MODEL # Added model parameter


class RegenerateRequest(BaseModel):
    conversation_id: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "rachel"  # Defaulting to Rachel


class IntentInfo(BaseModel):
    intent: str
    confidence: float
    sub_intents: List[str]
    action_type: str
    tools: List[str]


class FileAnalysisResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    files: List[Dict[str, Any]] = []
    truncated: bool = False


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
                logger.warning(f"{description} encountered transient error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            else:
                logger.error(f"{description} failed: {e}")
                break
    
    if last_exception:
        raise last_exception


async def get_user(
    request: Request,
    response: Response,
    remember: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Production-grade user recognition with:
    - Session token validation
    - Multi-cookie fallback strategy
    - Device fingerprinting
    - Session refresh logic
    - Automatic session creation
    """
    await cleanup_session_cache()
    
    # Get all cookie values
    primary_id = request.cookies.get(PRIMARY_COOKIE)
    backup_id = request.cookies.get(BACKUP_COOKIE)
    device_cookie = request.cookies.get(DEVICE_COOKIE)
    stored_fingerprint = request.cookies.get(FINGERPRINT_COOKIE)
    session_token = request.cookies.get(SESSION_TOKEN_COOKIE)
    session_expiry = request.cookies.get(SESSION_EXPIRY_COOKIE)
    
    # Generate current fingerprint
    current_fingerprint = generate_device_fingerprint(request)
    
    # Determine remember preference
    if remember is None:
        # Default to True unless session is about to expire
        remember = not is_session_expired(session_expiry or "0")
    
    user_obj = {
        "id": None,
        "email": None,
        "memory": "",
        "fingerprint": current_fingerprint,
        "session_valid": False,
        "session_token": None
    }

    # Priority 1: Validate existing session token
    user_id = None
    if primary_id and session_token:
        # Check if session is expired
        if is_session_expired(session_expiry or "0"):
            logger.info(f"Session expired for user {primary_id[:8]}...")
            clear_session_cookies(response)
        else:
            # Validate token
            token_valid = await validate_session_token(primary_id, session_token)
            if token_valid:
                user_id = primary_id
                user_obj["session_valid"] = True
                user_obj["session_token"] = session_token
                
                # Check if session should be refreshed
                if should_refresh_session(session_expiry or "0"):
                    logger.info(f"Refreshing session for user {user_id[:8]}...")
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
            else:
                logger.warning(f"Invalid session token for user {primary_id[:8]}...")
    
    # Priority 2: Try backup cookie
    if not user_id and backup_id:
        user_id = backup_id
        logger.info(f"User recovered via backup cookie: {user_id[:8]}...")
    
    # Priority 3: Try device fingerprint lookup
    if not user_id and device_cookie:
        try:
            fp_part = device_cookie.split("_")[0] if "_" in device_cookie else device_cookie
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1),
                description="User Lookup by Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via device fingerprint: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Fingerprint lookup failed: {e}")

    # Priority 4: Try stored fingerprint
    if not user_id and stored_fingerprint:
        try:
            fp_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", stored_fingerprint).limit(1),
                description="User Lookup by Stored Fingerprint"
            )
            if fp_resp.data:
                user_id = fp_resp.data[0]["id"]
                logger.info(f"User recovered via stored fingerprint: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Stored fingerprint lookup failed: {e}")

    # Load user data if we found an ID
    if user_id:
        try:
            user_resp = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", user_id).limit(1),
                description="User Lookup by ID"
            )
            if user_resp.data:
                u = user_resp.data[0]
                user_obj = {
                    "id": u["id"],
                    "email": u.get("email"),
                    "memory": u.get("memory", ""),
                    "is_premium": u.get("is_premium", False),
                    "is_lifetime": u.get("is_lifetime", False),
                    "plan": u.get("plan", "free"),
                    "fingerprint": current_fingerprint,
                    "session_valid": user_obj.get("session_valid", False),
                    "session_token": user_obj.get("session_token")
                }
                
                # Update fingerprint if changed
                if u.get("fingerprint") != current_fingerprint:
                    try:
                        await _execute_supabase_with_retry(
                            supabase.table("users").update({"fingerprint": current_fingerprint}).eq("id", user_id),
                            description="Update Fingerprint"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update fingerprint: {e}")
                
                # Create or refresh session
                if not user_obj["session_valid"]:
                    new_token = await create_user_session(user_id, current_fingerprint, remember)
                    user_obj["session_token"] = new_token
                    user_obj["session_valid"] = True
                
                # Set all cookies
                set_session_cookies(response, user_id, current_fingerprint, user_obj["session_token"], remember)
                
                logger.info(f"User authenticated: {user_id[:8]}... session_valid={user_obj['session_valid']}")
                return user_obj
        except Exception as e:
            logger.error(f"User data fetch failed: {e}")

    # Create new anonymous user
    new_id = str(uuid.uuid4())
    
    try:
        new_user_data = {
            "id": new_id,
            "email": f"anon+{new_id[:8]}@local",
            "memory": "",
            "fingerprint": current_fingerprint
        }
        
        await _execute_supabase_with_retry(
            supabase.table("users").upsert(new_user_data, on_conflict="id"),
            description="Create Anonymous User"
        )
        
        user_obj["id"] = new_id
        
    except Exception as e:
        logger.error(f"Failed to create anonymous user: {e}")
        user_obj["id"] = new_id

    # Create session for new user
    new_token = await create_user_session(new_id, current_fingerprint, remember)
    user_obj["session_token"] = new_token
    user_obj["session_valid"] = True
    
    # Set all cookies
    set_session_cookies(response, new_id, current_fingerprint, new_token, remember)
    
    logger.info(f"New user created: {new_id[:8]}... with fingerprint {current_fingerprint[:8]}...")
    
    return user_obj

async def update_user_memory(user_id: str, new_memory: str):
    """Update the user's long-term memory in the database."""
    try:
        await _execute_supabase_with_retry(
            supabase.table("users").update({"memory": new_memory}).eq("id", user_id),
            description="Update User Memory"
        )
        # Update cache
        if user_id in _session_cache:
            _session_cache[user_id]["memory"] = new_memory
    except Exception as e:
        logger.error(f"Failed to update user memory: {e}")


def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}


def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def get_elevenlabs_headers():
    return {"xi-api-key": ELEVENLABS_API_KEY}


# =========================
# EXTERNAL TOOLS IMPLEMENTATIONS
# =========================
async def solve_math(query: str) -> str:
    """Primary Math Solver using Wolfram Alpha, fallback to LLM"""
    if wolframalpha and WOLFRAM_ALPHA_API_KEY:
        try:
            client = wolframalpha.Client(WOLFRAM_ALPHA_API_KEY)
            res = client.query(query)
            if hasattr(res, 'results') and res.results:
                return next(res.results).text
            elif hasattr(res, 'pods') and len(res.pods) > 1:
                return res.pods[1].text if res.pods[1].text else "Wolfram could not solve this."
            else:
                return "Wolfram could not solve this."
        except Exception as e:
            logger.warning(f"Wolfram failed: {e}, falling back to LLM")
    
    # Fallback to LLM
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": DEFAULT_LLM_MODEL, "messages": [{"role": "user", "content": f"Solve step by step: {query}"}]}
        )
        return r.json()["choices"][0]["message"]["content"]


async def search_google(query: str) -> str:
    """Search Google using SerpApi"""
    if not SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY not configured. Skipping web search.")
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": 5  # Top 5 results
            }
            response = await client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Format results
            results = data.get("organic_results", [])
            formatted_results = []
            for res in results[:5]: # Limit to top 5
                title = res.get("title", "N/A")
                link = res.get("link", "#")
                snippet = res.get("snippet", "")
                formatted_results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
            
            return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"SerpApi search failed: {e}")
        return ""


JUDGE0_LANGUAGES = {
    "python": 71, "javascript": 63, "java": 62, "cpp": 54,
    "c": 50, "csharp": 51, "go": 60, "rust": 73, "sql": 82
}

async def execute_code(code: str, language: str) -> dict:
    if not JUDGE0_API_KEY:
        return {"error": "Judge0 API Key not configured"}

    lang_id = JUDGE0_LANGUAGES.get(language.lower(), 71)
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Submit
        r = await client.post(
            "https://judge0-ce.p.rapidapi.com/submissions",
            headers={"X-RapidAPI-Key": JUDGE0_API_KEY, "Content-Type": "application/json"},
            json={"source_code": code, "language_id": lang_id, "stdin": ""}
        )
        if r.status_code != 201:
             return {"error": f"Judge0 Submit Failed: {r.status_code}"}
        
        token = r.json()["token"]
        
        # Poll
        for _ in range(10):
            await asyncio.sleep(1)
            r = await client.get(
                f"https://judge0-ce.p.rapidapi.com/submissions/{token}",
                headers={"X-RapidAPI-Key": JUDGE0_API_KEY}
            )
            data = r.json()
            if data["status"]["id"] in (1, 2): continue # Processing
            return {
                "stdout": data.get("stdout", ""),
                "stderr": data.get("stderr", ""),
                "status": data["status"]["description"]
            }
    return {"error": "Execution timed out"}


# =========================
# CORE CHAT LOGIC
# =========================
async def stream_groq_chat(messages: list, model: str = DEFAULT_LLM_MODEL, max_tokens: int = 1024):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
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
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except:
                            pass
    except httpx.ConnectError:
        logger.error("Failed to connect to Groq API.")
        raise Exception("Connection to AI service failed.")
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        raise e


async def get_history(conv_id: str) -> list:
    """Fetch conversation history efficiently"""
    try:
        result = await _execute_supabase_with_retry(
            supabase.table("messages")
            .select("role, content")
            .eq("conversation_id", conv_id)
            .order("created_at", desc=False) # Oldest first
            .limit(20), # Last 20 messages context
            description="Fetch History"
        )
        if result.data:
            return [{"role": m["role"], "content": m["content"]} for m in result.data]
        return []
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        return []

async def save_message(user_id: str, conv_id: str, role: str, content: str):
    """Helper to save a single message"""
    try:
        await _execute_supabase_with_retry(
            supabase.table("messages").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": conv_id,
                "role": role,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Save Message"
        )
    except Exception as e:
        logger.error(f"Failed to save message: {e}")


# =========================
# ENDPOINTS
# =========================

@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")
    
    body = {}
    remember = True
    
    if "application/json" in content_type:
        try:
            body = await req.json()
            remember = body.get("remember", True)
        except Exception:
            raise HTTPException(400, "Invalid JSON")
    elif "multipart/form-data" in content_type:
        try:
            form = await req.form()
            body = dict(form)
            remember = body.get("remember", True)
        except Exception:
            raise HTTPException(400, "Invalid Form Data")
    else:
        raise HTTPException(415, f"Unsupported content-type: {content_type}")

    user = await get_user(req, res, remember=remember)

    # Handle File Upload immediately if present (for Form Data)
    if "multipart/form-data" in content_type and "file" in body:
        file: UploadFile = body["file"]
        content = await file.read()

        logger.info(f"File upload: {file.filename} ({format_file_size(len(content))})")

        if file.content_type and file.content_type.startswith("image/"):
            return JSONResponse({"error": "Image analysis endpoint not implemented in this snippet"}, status_code=501)
        
        result = await extract_file_content(content, file.filename)
        return JSONResponse(content=result.to_dict())

    # Proceed with Chat Logic
    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    
    # RESOLVE MODEL NAME (THE FIX)
    # Get the model name from the request (e.g., "helox")
    request_model = body.get("model")
    # Map it to the actual API model ID using the new resolve_model function
    api_model_name = resolve_model(request_model)

    if not prompt:
        raise HTTPException(400, "Prompt required")

    conversation_exists = False
    
    if conv_id:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", conv_id).limit(1),
            description="Check Conversation Existence"
        )
        if check.data:
            conversation_exists = True
        else:
            logger.warning(f"Conversation {conv_id} not found in DB. Creating new conversation.")
            conv_id = str(uuid.uuid4())

    if not conv_id:
        conv_id = str(uuid.uuid4())

    if not conversation_exists:
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": conv_id, 
                "user_id": user["id"],
                "title": prompt[:30] if prompt else "New Chat", 
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create Conversation"
        )

    try:
        await save_message(user["id"], conv_id, "user", prompt)
    except Exception as e:
        logger.error(f"Failed to save user message: {e}")

    intent_result = detect_intent(prompt)
    action_type = get_action_type(prompt)
    required_tools = get_required_tools(prompt)

    if intent_result:
        logger.info(
            f"[INTENT] action={action_type.value} "
            f"intent={intent_result.intent.value} "
            f"confidence={intent_result.confidence:.2%} "
            f"sub_intents={[i.value for i in intent_result.sub_intents]} "
            f"tools={required_tools}"
        )
    else:
        logger.info(f"[INTENT] action={action_type.value} (no specific intent)")

    # 1. MATH (Wolfram Alpha)
    if intent_result and intent_result.intent == IntentCategory.MATHEMATICAL:
        async def math_gen():
            yield sse({"type": "status", "message": "Calculating with Wolfram Alpha..."})
            try:
                result = await solve_math(prompt)
                # Simulate streaming for math result
                for char in result:
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.01)
                yield sse({"type": "done"})
            except Exception as e:
                yield sse({"type": "error", "message": str(e)})
        return StreamingResponse(math_gen(), media_type="text/event-stream")

    # 2. CODE EXECUTION (Judge0)
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

    # 3. RESEARCH (SerpApi)
    if intent_result and intent_result.intent == IntentCategory.RESEARCH:
        async def research_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            
            try:
                yield sse({"type": "status", "message": "Searching the web..."})
                search_results = await search_google(prompt)
                
                if not search_results:
                    # Fallback if search fails or no key
                    yield sse({"type": "token", "text": "I couldn't perform a web search right now, but I'll try to answer from my knowledge.\n\n"})
                    # Continue with default chat logic
                    history = await get_history(conv_id) if conv_id else []
                    base_system = get_system_prompt(prompt) + "\n\nYou are a helpful research assistant."
                    user_memory = user.get("memory", "")
                    if user_memory:
                        base_system += f"\n\nUser Context: {user_memory}"
                    full_history = [{"role": "system", "content": base_system}] + history
                    
                    full_text = ""
                    async for token in stream_groq_chat(full_history):
                        if task.cancelled():
                            break
                        full_text += token
                        yield sse({"type": "token", "text": token})
                        
                    # Memory Update & Save
                    new_memory = (user_memory + "\n" + full_text[-500:]) if user_memory else full_text[-1000:]
                    if len(new_memory) > 5000:
                        new_memory = new_memory[-5000:]
                    asyncio.create_task(update_user_memory(user["id"], new_memory))
                    await save_message(user["id"], conv_id, "assistant", full_text)
                else:
                    # Feed search results to LLM
                    system_prompt = get_system_prompt(prompt) + """
You are a research assistant. The user has asked a question that requires current information.
I have performed a web search and gathered the following results for you.

Search Results:
""" + search_results + """

Please answer the user's question based on these results. If the results don't contain enough information, say so. Cite the sources if possible."""
                    
                    user_memory = user.get("memory", "")
                    if user_memory:
                        system_prompt += f"\n\nUser Context: {user_memory}"
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    full_text = ""
                    async for token in stream_groq_chat(messages): # <--- Uses mapped model if default
                        if task.cancelled():
                            break
                        full_text += token
                        yield sse({"type": "token", "text": token})
                    
                    # Memory Update & Save
                    new_memory = (user_memory + "\n" + full_text[-500:]) if user_memory else full_text[-1000:]
                    if len(new_memory) > 5000:
                        new_memory = new_memory[-5000:]
                    asyncio.create_task(update_user_memory(user["id"], new_memory))
                    await save_message(user["id"], conv_id, "assistant", full_text)
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Research Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred during research."})
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(research_gen(), media_type="text/event-stream")

    # 4. DEFAULT CHAT (Gemma 2 27B)
    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task

            try:
                try:
                    history = await get_history(conv_id) if conv_id else []
                except Exception as e:
                    logger.error(f"History fetch failed: {e}")
                    history = [] 
                
                base_system = get_system_prompt(prompt)
                user_memory = user.get("memory", "")
                if user_memory:
                    base_system += f"\n\nUser Context: {user_memory}"
                
                full_history = [{"role": "system", "content": base_system}] + history
                
                full_text = ""
                async for token in stream_groq_chat(full_history, model=api_model_name): # <--- Use resolved model
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})

                # MEMORY UPDATE LOGIC FOR GENERAL CHAT
                new_memory = (user_memory + "\n" + full_text[-500:]) if user_memory else full_text[-1000:]
                if len(new_memory) > 5000:
                    new_memory = new_memory[-5000:]
                
                asyncio.create_task(update_user_memory(user["id"], new_memory))

                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")
                
                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Universal Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred processing your request."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        # Non-streaming fallback
        history = await get_history(conv_id)
        base_system = get_system_prompt(prompt)
        user_memory = user.get("memory", "")
        if user_memory:
            base_system += f"\n\nUser Context: {user_memory}"
            
        full_history = [{"role": "system", "content": base_system}] + history
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": api_model_name, "messages": full_history, "max_tokens": 1024}  # Use resolved model
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            
            # Update memory
            new_memory = (user_memory + "\n" + reply[-500:]) if user_memory else reply[-1000:]
            if len(new_memory) > 5000:
                new_memory = new_memory[-5000:]
            asyncio.create_task(update_user_memory(user["id"], new_memory))
            
            await save_message(user["id"], conv_id, "assistant", reply)
            
            return {"reply": reply}


# =========================
# ENHANCED FILE ANALYSIS ENDPOINT
# =========================
@app.post("/analyze/file")
async def analyze_file(
    req: Request,
    file: UploadFile = File(...),
    stream: bool = True
):
    """Enhanced file analysis endpoint supporting:
    - All code files (.py, .js, .ts, .java, .cpp, .go, .rs, etc.)
    - Archives (.zip, .tar, .gz, etc.) - extracts and analyzes contents
    - Documents (.pdf, .txt, .md, .csv, etc.)
    - Images (visual analysis)
    - Large files (up to 50MB single,100MB archives)
    """
    user = await get_user(req, Response())
    
    # Read file content
    content = await file.read()
    filename = file.filename or "unknown"
    content_type = file.content_type or ""
    file_size = len(content)
    
    logger.info(f"[FILE] Upload: {filename} ({format_file_size(file_size)}, type={content_type})")
    
    if not content:
        raise HTTPException(400, "Empty file")
    
    # Check file size limits
    category = get_file_category(filename)
    max_allowed = MAX_ZIP_SIZE if category == FileCategory.ARCHIVE else MAX_FILE_SIZE
    
    if file_size > max_allowed:
        raise HTTPException(
            400, 
            f"File too large ({format_file_size(file_size)}). Maximum: {format_file_size(max_allowed)}"
        )
    
    # Handle images separately
    if content_type.startswith("image/") or category == FileCategory.IMAGE:
        return await handle_image_analysis(content, filename, stream)
    
    # Extract content from file
    result = await extract_file_content(content, filename)
    
    logger.info(
        f"[FILE] Extracted: {filename} -> "
        f"category={result.metadata.get('category')}, "
        f"truncated={result.truncated}, "
        f"content_len={len(result.content)}"
    )
    
    # Handle archives with multiple files
    if category == FileCategory.ARCHIVE and result.files:
        return await handle_archive_analysis(result, stream)
    
    # Handle single file analysis
    return await handle_text_analysis(result, stream, file_metadata=result.metadata)


async def handle_archive_analysis(
    result: FileExtractionResult,
    stream: bool
):
    """Special handling for archive files with multiple extracted files"""
    
    # Build a comprehensive analysis prompt
    files_summary = []
    code_files = []
    text_files = []
    
    for f in result.files:
        status = f.get("status", "unknown")
        if status == "extracted":
            files_summary.append(f"- {f['name']} ({f.get('size_formatted', '?')}) - {f.get('category', '?')}")
            if f.get("category") == "code":
                code_files.append(f)
            else:
                text_files.append(f)
        elif status in ("binary", "media"):
            files_summary.append(f"- {f['name']} ({f.get('size_formatted', '?')}) - {status}")
        elif status == "skipped":
            files_summary.append(f"- {f['name']} - skipped: {f.get('reason', '?')}")
        else:
            files_summary.append(f"- {f['name']} - {status}")
    
    summary_intro = f"""Archive Analysis: {result.metadata.get('filename', 'unknown')}
Total entries: {result.metadata.get('entry_count', 0)}
Processed: {result.metadata.get('processed_count', 0)}
Text files extracted: {result.metadata.get('extracted_count', 0)}

Files found:
{chr(10).join(files_summary)}

"""

    full_content = summary_intro + result.content
    
    messages = [
        {
            "role": "system",
            "content": get_system_prompt("") + """

You are analyzing an archive file (ZIP, TAR, etc.). The archive contents have been extracted and provided below.

Your task:
1. Provide an overview of what this archive contains
2. Identify the main purpose/type of the project or files
3. If it's a code project, describe the structure, technologies used, and main functionality
4. Highlight any important files or configurations
5. Note any potential issues, missing files, or areas of concern
6. If appropriate, provide a summary of the code functionality

Be organized and clear in your analysis."""
        },
        {
            "role": "user",
            "content": full_content
        }
    ]

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                # First, send metadata
                yield sse({
                    "type": "file_metadata",
                    "metadata": result.metadata,
                    "files": result.files
                })
                
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    yield sse({"type": "token", "text": token})
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Archive analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": DEFAULT_LLM_MODEL, "messages": messages}
        )
        r.raise_for_status()

    return {
        "analysis": r.json()["choices"][0]["message"]["content"],
        "metadata": result.metadata,
        "files": result.files
    }


async def handle_text_analysis(
    result: FileExtractionResult,
    stream: bool,
    file_metadata: Dict[str, Any] = None
):
    """Handle single text/code file analysis via LLM"""
    if file_metadata is None: file_metadata = result.metadata
    
    content = result.content
    filename = result.metadata.get("filename", "file.txt")
    
    # Prepare prompt for LLM to analyze the file
    prompt = f"""I have uploaded a file named '{filename}'.
Here is its content:

{content}

Please analyze this file. If it is code, review it for errors or improvements. If it is a document, summarize it. If it is data, describe its structure."""

    messages = [
        {
            "role": "system",
            "content": get_system_prompt("") + "You are a helpful coding and document analysis assistant. Provide clear, actionable insights on uploaded files."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if stream:
        async def gen():
            task = asyncio.current_task()
            try:
                # Send metadata first
                yield sse({
                    "type": "file_metadata",
                    "metadata": file_metadata,
                    "files": []
                })
                
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Text analysis stream error: {e}")
                yield sse({"type": "error", "message": "Analysis failed."})
        
        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        # Non-streaming response
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": DEFAULT_LLM_MODEL, "messages": messages}
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            
            return {
                "analysis": reply,
                "metadata": file_metadata,
                "content": result.content # returning raw content as well for frontend use
            }


async def handle_image_analysis(
    content: bytes,
    filename: str,
    stream: bool
):
    """Stub for image analysis - in a real env, this would call OpenAI Vision"""
    metadata = {
        "filename": filename,
        "category": "image",
        "size": len(content)
    }
    
    if stream:
        async def gen():
            yield sse({
                "type": "file_metadata",
                "metadata": metadata,
                "files": []
            })
            # In a real implementation, you would stream the analysis here.
            yield sse({"type": "token", "text": f"I see an image named {filename}. However, visual analysis is not currently enabled in this backend configuration."})
            yield sse({"type": "done"})
        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        return JSONResponse(content={
            "content": f"[Image: {filename}]",
            "metadata": metadata,
            "error": "Vision not enabled"
        })


@app.get("/file-types")
async def get_supported_file_types():
    """Return list of supported file types"""
    return {
        "code": sorted(list(CODE_EXTENSIONS)),
        "document": sorted(list(DOCUMENT_EXTENSIONS)),
        "data": sorted(list(DATA_EXTENSIONS)),
        "image": sorted(list(IMAGE_EXTENSIONS)),
        "audio": sorted(list(AUDIO_EXTENSIONS)),
        "video": sorted(list(VIDEO_EXTENSIONS)),
        "archive": sorted(list(ARCHIVE_EXTENSIONS)),
        "limits": {
            "max_file_size": format_file_size(MAX_FILE_SIZE),
            "max_zip_size": format_file_size(MAX_ZIP_SIZE),
            "max_zip_entries": MAX_ZIP_ENTRIES,
            "max_extracted_size": format_file_size(MAX_EXTRACTED_SIZE),
            "max_text_length": format_file_size(MAX_TEXT_LENGTH)
        }
    }


# =========================
# SESSION MANAGEMENT ENDPOINTS
# =========================
@app.post("/session/validate")
async def validate_session(req: Request, res: Response):
    """Validate current session and return user info"""
    user = await get_user(req, res)
    return {
        "valid": user.get("session_valid", False),
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_authenticated": bool(user.get("email") and not user["email"].startswith("anon+"))
    }


@app.post("/session/refresh")
async def refresh_session(req: Request, res: Response):
    """Manually refresh the current session"""
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}
    remember = body.get("remember", True)
    
    user = await get_user(req, res, remember=remember)
    
    # Force session refresh
    new_token = await create_user_session(
        user["id"],
        user.get("fingerprint", ""),
        remember
    )
    
    set_session_cookies(res, user["id"], user.get("fingerprint", ""), new_token, remember)
    
    return {
        "status": "refreshed",
        "user_id": user["id"],
        "expires_in": SESSION_DURATION if remember else 24 * 60 * 60
    }


@app.post("/session/logout")
async def logout(req: Request, res: Response):
    """Logout and clear all session data"""
    user_id = req.cookies.get(PRIMARY_COOKIE)
    
    if user_id:
        try:
            # Invalidate all sessions for this user
            await _execute_supabase_with_retry(
                supabase.table("user_sessions")
                .update({"is_valid": False})
                .eq("user_id", user_id),
                description="Invalidate User Sessions"
            )
        except Exception as e:
            logger.error(f"Failed to invalidate sessions: {e}")
        
        # Clear cache
        if user_id in _session_cache:
            del _session_cache[user_id]
    
    clear_session_cookies(res)
    
    return {"status": "logged_out"}


# =========================
# ELEVENLABS & CHAT UTILS ENDPOINTS
# =========================
@app.get("/tts/voices")
async def get_voices(request: Request, response: Response):
    """Fetch available voices from ElevenLabs"""
    try:
        headers = get_elevenlabs_headers()
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.elevenlabs.io/v1/voices", headers=headers)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"ElevenLabs Voices Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch voices.")
    except Exception as e:
        logger.error(f"Voices Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Text-to-Speech using ElevenLabs"""
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API Key not configured.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{req.voice}"
        headers = get_elevenlabs_headers()
        
        # Payload for ElevenLabs
        payload = {
            "text": req.text,
            "model_id": "eleven_multilingual_v2",
            "output_format": "mp3_44100_128"
        }

        async def stream_audio():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_audio(), media_type="audio/mpeg")

    except httpx.HTTPStatusError as e:
        logger.error(f"TTS Error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="TTS failed.")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Speech-to-Text (STT) using ElevenLabs Scribe"""
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API Key not configured.")

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        
        # ElevenLabs STT requires form-data with 'file' and 'model_id'
        data = {"model_id": "scribe_v1"}
        
        # Read file content
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                url, 
                headers=headers, 
                data=data, 
                files={"file": (file.filename, file_content, file.content_type)}
            )
            r.raise_for_status()
            return r.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"STT Error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail="Speech recognition failed.")
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/regenerate")
async def regenerate_response(req: RegenerateRequest):
    """Regenerate the last assistant response for a conversation"""
    if not req.conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID required")

    try:
        # 1. Fetch history
        history = await get_history(req.conversation_id)
        
        # 2. If last message was from assistant, we conceptually 'pop' it (skip it)
        # NOTE: We do NOT delete it from DB. The frontend handles visual replacement.
        if history and history[-1]["role"] == "assistant":
            # Remove from local list to avoid sending it back to LLM
            history.pop()
        
        # 3. Find the last user message to use as prompt
        prompt = None
        for msg in reversed(history):
            if msg["role"] == "user":
                prompt = msg["content"]
                break
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No user message found to regenerate from.")

        # 4. Stream response using existing logic
        # We construct the history up to the last user message
        # (history currently contains msgs up to the last AI msg which we popped, so it ends at User msg)
        
        base_system = get_system_prompt(prompt)
        full_history = [{"role": "system", "content": base_system}] + history
        
        async def event_gen():
            try:
                full_text = ""
                # Use resolved model name
                async for token in stream_groq_chat(full_history, model=resolve_model(DEFAULT_LLM_MODEL)):
                    full_text += token
                    # Use SSE format expected by frontend
                    yield sse({"type": "token", "text": token})
                
                # Save the new assistant message
                # We save it as a new entry, effectively appending to the conversation
                # The frontend usually deletes the old AI message visually, leaving DB with: User... OldAI, NewAI
                await save_message("system", req.conversation_id, "assistant", full_text)
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Regenerate Error: {e}")
                yield sse({"type": "error", "message": "Regeneration failed."})

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Regenerate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/newchat")
async def create_new_chat():
    """Initialize a new chat session ID"""
    # Generate a new ID
    new_id = str(uuid.uuid4())
    
    # You could insert a 'conversations' row here if desired
    # await _execute_supabase_with_retry(...)
    
    return JSONResponse(content={"conversation_id": new_id})


# =========================
# SPECIALIZED HANDLERS
# =========================
async def handle_math_request(prompt: str, user: Dict[str, Any], conv_id: str, stream: bool):
    system_prompt = get_system_prompt(prompt) + """

You are also a mathematical expert. When solving math problems:
1. Show your work step-by-step
2. Explain each step clearly
3. Use proper mathematical notation
4. Verify your answer
5. If it's a proof, be rigorous

Format complex equations clearly using LaTeX-style notation where appropriate."""

    user_memory = user.get("memory", "")
    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    history = await get_history(conv_id) if conv_id else []
    messages = [{"role": "system", "content": system_prompt}] + history

    if stream:
        async def gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                async for token in stream_groq_chat(messages):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                # MEMORY UPDATE
                new_memory = (user_memory + "\n" + full_text[-500:]) if user_memory else full_text[-1000:]
                asyncio.create_task(update_user_memory(user["id"], new_memory))

                try:
                    await save_message(user["id"], conv_id, "assistant", full_text)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

                yield sse({"type": "done"})
            
            except Exception as e:
                logger.error(f"Math Stream Error: {e}")
                yield sse({"type": "error", "message": "An error occurred."})
            
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(gen(), media_type="text/event-stream")
    
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": resolve_model(DEFAULT_LLM_MODEL), "messages": messages, "max_tokens": 2048}
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    # MEMORY UPDATE
    new_memory = (user_memory + "\n" + reply[-500:]) if user_memory else reply[-1000:]
    asyncio.create_task(update_user_memory(user["id"], new_memory))

    await save_message(user["id"], conv_id, "assistant", reply)
    return {"reply": reply}


@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "services": {
            "groq_gemma2_27b": bool(GROQ_API_KEY),
            # Legacy Key for Frontend compatibility
            "groq_llama405b": bool(GROQ_API_KEY), 
            
            "judge0": bool(JUDGE0_API_KEY), 
            "wolfram": bool(WOLFRAM_ALPHA_API_KEY),
            "serpapi": bool(SERPAPI_API_KEY) # Added SerpApi health check
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
