# ============================
# main.py — Zynara Ultra v5 (PART 1/3)
# Hybrid architecture (local GPUs + cloud fallback)
# Core imports, config, utilities, moderation, model routing, RAG helpers, streaming
# ============================

import os
import io
import sys
import json
import time
import uuid
import base64
import hashlib
import logging
import asyncio
import traceback
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional heavy libs — import safely and fallback
try:
    import torch
except Exception:
    torch = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
except Exception:
    StableDiffusionPipeline = None
    StableDiffusionInpaintPipeline = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
except Exception:
    connections = Collection = FieldSchema = CollectionSchema = DataType = None

try:
    import ffmpeg
except Exception:
    ffmpeg = None

try:
    import redis as redis_lib
except Exception:
    redis_lib = None

try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

# ---------------- logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zynara")

# ---------------- config ----------------
APP_NAME = os.getenv("APP_NAME", "Zynara Ultra v5")
PORT = int(os.getenv("PORT", "7860"))
HF_TOKEN = os.getenv("HF_TOKEN")                 # Hugging Face Inference or model hosting token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     # optional (moderation / OpenAI endpoints)
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")     # ElevenLabs TTS
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_URL = os.getenv("REDIS_URL")
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "1") == "1"
VLLM_URL = os.getenv("VLLM_URL")                 # e.g. http://vllm:8000 or TGI endpoint for streaming
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
TMP_DIR = os.getenv("TMP_DIR", "/tmp/zynara")
os.makedirs(TMP_DIR, exist_ok=True)
MEDIA_DIR = os.path.join(TMP_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# Hybrid toggles
USE_LOCAL_SDXL = os.getenv("USE_LOCAL_SDXL", "1") == "1"
USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "1") == "1"
USE_LLAMA_405B = os.getenv("USE_LLAMA_405B", "0") == "1"

# ---------------- app ----------------
app = FastAPI(title=APP_NAME)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- optional clients ----------------
supabase = None
if create_supabase_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.warning("Supabase init failed: %s", e)

redis_client = None
if redis_lib and REDIS_URL:
    try:
        redis_client = redis_lib.from_url(REDIS_URL)
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis init failed: %s", e)

# ---------------- utilities ----------------
def now_ts() -> int:
    return int(time.time())

def write_temp_file(data: bytes, suffix: str = "") -> str:
    path = os.path.join(MEDIA_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------------- moderation ----------------
def moderate_text_basic(text: str) -> Tuple[bool, Optional[str]]:
    if not text:
        return True, None
    banned = ["bomb", "explode", "kill", "terror", "suicide"]
    for b in banned:
        if b in text.lower():
            return False, f"Blocked word: {b}"
    # OpenAI moderation if available
    if OPENAI_API_KEY and httpx:
        try:
            resp = httpx.post("https://api.openai.com/v1/moderations", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, json={"input": text}, timeout=10.0)
            if resp.status_code == 200:
                jr = resp.json()
                if jr.get("results") and jr["results"][0].get("flagged"):
                    return False, "OpenAI moderation flagged content"
        except Exception:
            logger.debug("OpenAI moderation API call failed", exc_info=True)
    return True, None

# ---------------- model registry & cache ----------------
MODEL_REGISTRY: Dict[str, List[str]] = {
    "text:chat": ["meta-llama/Llama-3-70B", "tiiuae/falcon-180b"],
    "code:gen": ["bigcode/starcoder", "Salesforce/codegen-6B-multi"],
    "image:sdxl": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "image:dalle3": ["openai/dall-e-3"],  # conceptual placeholder
    "vision:vqa": ["Salesforce/blip-vqa-large"],
    "speech:stt": ["openai/whisper-large-v2"],
    "audio:tts": ["elevenlabs"],
    "text:embed": [EMBED_MODEL_NAME],
    "text:summarization": ["facebook/bart-large-cnn"],
}

MODEL_CACHE: Dict[str, Any] = {}

def get_best_model_id(key: str) -> Optional[str]:
    lst = MODEL_REGISTRY.get(key, [])
    return lst[0] if lst else None

# ---------------- lazy loaders ----------------
def lazy_load_text_local(preferred: List[str]):
    key = "text:local"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key], MODEL_CACHE.get(f"{key}:id")
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, preferred[0] if preferred else None
    for mid in preferred:
        if "405b" in mid.lower() and not USE_LLAMA_405B:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")
            MODEL_CACHE[key] = (tok, model)
            MODEL_CACHE[f"{key}:id"] = mid
            logger.info("Loaded local model %s", mid)
            return (tok, model), mid
        except Exception:
            logger.exception("Failed to load local text model %s", exc_info=True)
            continue
    return None, preferred[0] if preferred else None

def lazy_load_sdxl_local():
    key = "sdxl"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    if StableDiffusionPipeline is None or torch is None:
        return None
    model_id = get_best_model_id("image:sdxl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        if device == "cuda":
            pipe = pipe.to("cuda")
        MODEL_CACHE[key] = pipe
        logger.info("Loaded SDXL local pipeline")
        return pipe
    except Exception:
        logger.exception("Failed to load SDXL", exc_info=True)
        return None

def lazy_load_whisper_local():
    key = "whisper"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    if WhisperModel is None:
        return None
    try:
        w = WhisperModel(get_best_model_id("speech:stt") or "openai/whisper-large-v2")
        MODEL_CACHE[key] = w
        logger.info("Loaded Whisper local model")
        return w
    except Exception:
        logger.exception("Failed to load Whisper", exc_info=True)
        return None

# ---------------- HF inference helpers ----------------
def hf_inference_request(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    if httpx is None:
        raise RuntimeError("httpx not available")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    if isinstance(inputs, (bytes, bytearray)):
        r = httpx.post(url, headers=headers, content=inputs, timeout=timeout)
    else:
        body = {"inputs": inputs}
        if params:
            body["parameters"] = params
        r = httpx.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

async def hf_inference_async(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    if httpx is None:
        raise RuntimeError("httpx not available")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        if isinstance(inputs, (bytes, bytearray)):
            resp = await client.post(url, headers=headers, content=inputs)
        else:
            body = {"inputs": inputs}
            if params:
                body["parameters"] = params
            resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()

# ---------------- RAG (Milvus) helpers ----------------
_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL

def init_milvus():
    if connections is None:
        logger.debug("pymilvus not installed — skipping Milvus init")
        return False
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        try:
            existing = Collection.list()
        except Exception:
            existing = []
        if "zynara_embeddings" not in [c.name for c in existing]:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="meta", dtype=DataType.VARCHAR, max_length=4096),
            ]
            schema = CollectionSchema(fields, description="zynara embeddings")
            Collection("zynara_embeddings", schema)
        logger.info("Milvus initialized")
        return True
    except Exception:
        logger.exception("Milvus init failed")
        return False

def rag_upsert(doc_id: str, text: str, meta: dict = None):
    if Collection is None:
        raise RuntimeError("milvus not available")
    col = Collection("zynara_embeddings")
    emb = get_embed_model().encode([text], convert_to_numpy=True)[0].astype("float32")
    meta_json = json.dumps(meta or {})
    col.insert([[doc_id], [emb.tolist()], [meta_json]])
    col.flush()

def rag_query(text: str, k: int = 5):
    if Collection is None:
        return []
    col = Collection("zynara_embeddings")
    emb = get_embed_model().encode([text], convert_to_numpy=True)[0].astype("float32")
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    res = col.search([emb.tolist()], "embedding", param=search_params, limit=k, output_fields=["meta"])
    out = []
    for r in res[0]:
        try:
            meta_field = json.loads(r.entity.get("meta")) if r.entity.get("meta") else None
        except Exception:
            meta_field = None
        out.append({"id": r.id, "score": r.distance, "meta": meta_field})
    return out

# ---------------- Rate limiting (in-memory, can back to Redis) ----------------
RATE_LIMIT: Dict[str, Tuple[float, float]] = {}
RATE_TOKENS = int(os.getenv("RATE_TOKENS", "120"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))

def consume_token(user_id: str, cost: int = 1) -> bool:
    now = time.time()
    tokens, last = RATE_LIMIT.get(user_id, (RATE_TOKENS, now))
    elapsed = now - last
    refill = (elapsed / RATE_WINDOW) * RATE_TOKENS
    tokens = min(RATE_TOKENS, tokens + refill)
    if tokens < cost:
        RATE_LIMIT[user_id] = (tokens, now)
        return False
    tokens -= cost
    RATE_LIMIT[user_id] = (tokens, now)
    return True

# ---------------- vLLM/TGI streaming client helper ----------------
async def stream_from_vllm(prompt: str, websocket: WebSocket, max_tokens: int = 512):
    """
    Connect to vLLM/TGI streaming endpoint (VLLM_URL) and forward tokens to the websocket client.
    Supports SSE-like or NDJSON streaming formats and naive fallback to text chunks.
    """
    if not VLLM_URL or httpx is None:
        raise RuntimeError("vLLM not configured or httpx missing")
    payload = {"prompt": prompt, "max_tokens": max_tokens, "stream": True}
    timeout = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{VLLM_URL}/v1/generate", json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                # Parse lines that may be "data: {...}" or JSON per-line
                for line in chunk.splitlines():
                    if not line.strip():
                        continue
                    try:
                        if line.startswith("data:"):
                            part = line[len("data:"):].strip()
                            if part == "[DONE]":
                                await websocket.send_json({"done": True})
                                return
                            obj = json.loads(part)
                            tok = obj.get("token") or obj.get("text") or obj.get("delta")
                            if tok:
                                await websocket.send_json({"delta": tok})
                            else:
                                # send entire object fallback
                                await websocket.send_json({"delta": obj})
                        else:
                            # try JSON
                            try:
                                obj = json.loads(line)
                                tok = obj.get("token") or obj.get("text") or obj.get("delta")
                                if tok:
                                    await websocket.send_json({"delta": tok})
                                else:
                                    await websocket.send_json({"delta": obj})
                            except Exception:
                                # plain text chunk
                                await websocket.send_json({"delta": line})
                    except Exception:
                        # best-effort forward
                        try:
                            await websocket.send_json({"delta": line})
                        except Exception:
                            pass
    try:
        await websocket.send_json({"done": True})
    except Exception:
        pass

# ---------------- Pydantic request models ----------------
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    model_hint: Optional[str] = "text:chat"

class ImageGenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    samples: int = 1
    model_hint: Optional[str] = "image:sdxl"

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    format: str = "mp3"

# ============================
# main.py — Zynara Ultra v5 (PART 2/3)
# Image / Video / Audio pipelines, TTS/STT, code-exec sandbox stubs, worker hooks
# ============================

# ---------------- Image generation & utilities ----------------
@app.post("/image/generate")
async def image_generate(req: ImageGenRequest):
    """
    Multi-backend image generation:
      - If model_hint contains 'dalle' or VLLM/OpenAI DALL·E3 available -> use HF/OpenAI endpoint
      - Else try local SDXL pipeline
      - Else fallback to HF SDXL inference
    Returns base64-encoded PNG images (list)
    """
    prompt = req.prompt
    model_hint = (req.model_hint or "image:sdxl").lower()

    # Prefer DALL·E3 via HF/OpenAI if explicitly requested
    if "dalle" in model_hint:
        # TODO: Replace with direct OpenAI DALL·E3 call when available
        model_id = get_best_model_id("image:dalle3") or get_best_model_id("image:sdxl")
        if HF_TOKEN and model_id:
            try:
                resp = await hf_inference_async(model_id, {"inputs": prompt, "parameters": {"width": req.width, "height": req.height, "num_images": req.samples}})
                return {"source": "hf", "result": resp}
            except Exception:
                logger.exception("HF DALL·E request failed", exc_info=True)

    # Try local SDXL
    if USE_LOCAL_SDXL:
        pipe = lazy_load_sdxl_local()
        if pipe:
            images_b64 = []
            for _ in range(max(1, req.samples)):
                try:
                    out = pipe(prompt, guidance_scale=7.5, num_inference_steps=28, height=req.height, width=req.width)
                    img = out.images[0]
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    images_b64.append(base64.b64encode(buf.read()).decode())
                except Exception:
                    logger.exception("Local SDXL generation failed", exc_info=True)
            if images_b64:
                return {"source": "sdxl_local", "images": images_b64}

    # HF fallback SDXL
    if HF_TOKEN:
        try:
            model_id = get_best_model_id("image:sdxl")
            if model_id:
                resp = await hf_inference_async(model_id, {"inputs": prompt, "parameters": {"width": req.width, "height": req.height, "num_images": req.samples}})
                return {"source": "hf", "result": resp}
        except Exception:
            logger.exception("HF SDXL fallback failed", exc_info=True)

    raise HTTPException(503, detail="No image generation backend available")

# ---------------- Image utilities: upscale, remove background, style transfer ----------------
@app.post("/image/upscale")
async def image_upscale(file: UploadFile = File(...)):
    """
    Upscale via HF Real-ESRGAN if available, else return error.
    """
    data = await file.read()
    tmp = write_temp_file(data, ".png")
    model_id = get_best_model_id("image:upscale")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp, "rb").read())
            return {"source": "hf", "result": resp}
        except Exception:
            logger.exception("HF upscale failed", exc_info=True)
    # TODO: integrate local Real-ESRGAN if installed
    raise HTTPException(503, detail="Upscale unavailable")

@app.post("/image/remove_bg")
async def image_remove_bg(file: UploadFile = File(...)):
    """
    Background removal using an inpainting model or external library.
    """
    data = await file.read()
    tmp = write_temp_file(data, ".png")
    model_id = get_best_model_id("image:inpaint")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp, "rb").read())
            return {"source": "hf", "result": resp}
        except Exception:
            logger.exception("HF inpaint failed", exc_info=True)
    # TODO: integrate rembg or MODNet locally
    raise HTTPException(503, detail="BG removal unavailable")

@app.post("/image/style_transfer")
async def image_style_transfer(image: UploadFile = File(...), style: str = Form(...)):
    """
    Style transfer using HF or placeholder.
    """
    data = await image.read()
    tmp = write_temp_file(data, ".png")
    model_id = get_best_model_id("image:style_transfer") or get_best_model_id("image:sdxl")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"image": open(tmp, "rb").read(), "style": style})
            return {"source": "hf", "result": resp}
        except Exception:
            logger.exception("HF style transfer failed", exc_info=True)
    raise HTTPException(503, detail="Style transfer unavailable")

# ---------------- TTS endpoint (ElevenLabs + fallback) ----------------
@app.post("/tts")
async def tts(req: TTSRequest):
    voice = (req.voice or "alloy").lower()
    if httpx and ELEVEN_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
            headers = {"xi-api-key": ELEVEN_API_KEY}
            payload = {"text": req.text}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                return StreamingResponse(io.BytesIO(r.content), media_type="audio/mpeg")
        except Exception:
            logger.exception("ElevenLabs TTS failed", exc_info=True)
    # TODO: integrate local TTS models (Coqui TTS / VITS) as fallback
    raise HTTPException(503, detail="TTS unavailable")

# ---------------- STT endpoint ----------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Accept audio file and return transcript.
    """
    data = await file.read()
    tmp = write_temp_file(data, ".wav")
    if USE_LOCAL_WHISPER:
        whisper = lazy_load_whisper_local()
        if whisper:
            try:
                segments, info = whisper.transcribe(tmp)
                text = " ".join([s.text for s in segments])
                return {"text": text}
            except Exception:
                logger.exception("Local whisper failed", exc_info=True)
    # HF fallback if configured
    if HF_TOKEN:
        try:
            stt_model = get_best_model_id("speech:stt")
            if stt_model:
                resp = hf_inference_request(stt_model, open(tmp, "rb").read())
                # normalization
                if isinstance(resp, dict) and "text" in resp:
                    return {"text": resp["text"]}
                if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "text" in resp[0]:
                    return {"text": resp[0]["text"]}
                return {"text": json.dumps(resp)}
        except Exception:
            logger.exception("HF STT failed", exc_info=True)
    raise HTTPException(503, detail="STT unavailable")

# ---------------- Code execution sandbox (secure) ----------------
# NOTE: This is a simplified, illustrative sandbox. For production you must use strong isolation
# (containers, seccomp, network disabled, resource limits, ephemeral VMs).
async def run_code_in_sandbox(code: str, language: str = "python", timeout: int = 20) -> Dict[str, Any]:
    """
    Execute code securely in an isolated environment (placeholder).
    For production: use firecracker, gVisor, docker-in-docker with strict limits.
    """
    # For Python only: naive exec with time/IO limits (DANGEROUS — demo only)
    if language.lower() in ("python", "py"):
        try:
            # Write to temporary file and run via subprocess in a docker container in real-world
            import subprocess, tempfile, shlex
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                tf.write(code)
                tf.flush()
                cmd = f"python {shlex.quote(tf.name)}"
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return {"success": False, "error": "timeout"}
                return {"success": proc.returncode == 0, "stdout": out.decode(errors="ignore"), "stderr": err.decode(errors="ignore")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        # TODO: integrate multi-language runners using ephemeral containers or cloud functions
        return {"success": False, "error": "language not supported in demo sandbox"}

@app.post("/code/run")
async def code_run(code: str = Form(...), language: str = Form("python")):
    res = await run_code_in_sandbox(code, language)
    return res

# ---------------- Video processing worker (detailed) ----------------
@app.post("/video/process")
async def video_process(file: UploadFile = File(...)):
    """
    Full video pipeline:
      - Save file
      - Extract audio (ffmpeg)
      - Transcribe (Whisper local or HF)
      - Extract keyframes (ffmpeg)
      - Optionally generate stylized frames (SDXL/DALL·E)
      - Summarize transcript (local HF)
    """
    data = await file.read()
    tmp_video = write_temp_file(data, ".mp4")
    try:
        # Extract audio
        audio_path = os.path.splitext(tmp_video)[0] + "_audio.wav"
        if ffmpeg:
            try:
                (
                    ffmpeg
                    .input(tmp_video)
                    .output(audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception:
                logger.exception("ffmpeg audio extraction failed", exc_info=True)
                audio_path = None
        else:
            audio_path = None

        # Transcribe audio
        transcript = ""
        if audio_path:
            whisper = lazy_load_whisper_local() if USE_LOCAL_WHISPER else None
            if whisper:
                try:
                    segments, info = whisper.transcribe(audio_path)
                    transcript = " ".join([s.text for s in segments])
                except Exception:
                    logger.exception("whisper transcribe failed", exc_info=True)
                    transcript = ""
            elif HF_TOKEN:
                try:
                    stt_model = get_best_model_id("speech:stt")
                    if stt_model:
                        resp = hf_inference_request(stt_model, open(audio_path, "rb").read())
                        if isinstance(resp, dict) and "text" in resp:
                            transcript = resp["text"]
                        elif isinstance(resp, list) and resp and isinstance(resp[0], dict) and "text" in resp[0]:
                            transcript = resp[0]["text"]
                        else:
                            transcript = json.dumps(resp)[:8000]
                except Exception:
                    logger.exception("HF STT failed", exc_info=True)

        # Extract keyframes (every 3 seconds)
        frames_dir = os.path.splitext(tmp_video)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        frames = []
        if ffmpeg:
            try:
                frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
                (
                    ffmpeg
                    .input(tmp_video)
                    .filter('fps', fps=1/3)
                    .output(frame_pattern, qscale=2)
                    .overwrite_output()
                    .run(quiet=True)
                )
                for fname in sorted(os.listdir(frames_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        frames.append(os.path.join(frames_dir, fname))
            except Exception:
                logger.exception("ffmpeg frame extraction failed", exc_info=True)

        # Generate stylized frames (limit)
        generated_frames = []
        if frames:
            pipe = lazy_load_sdxl_local()
            dalle_id = get_best_model_id("image:dalle3")
            for fpath in frames[:8]:
                try:
                    if pipe:
                        # Optionally use the original frame as conditioning (ControlNet/conditioning not implemented here)
                        prompt = f"Stylize this frame: {os.path.basename(fpath)}"
                        img = pipe(prompt, height=512, width=512).images[0]
                        b = io.BytesIO()
                        img.save(b, format="PNG")
                        b.seek(0)
                        generated_frames.append({"source": "sdxl", "frame": os.path.basename(fpath), "image_b64": base64.b64encode(b.read()).decode()})
                    elif HF_TOKEN and dalle_id:
                        resp = hf_inference_request(dalle_id, {"inputs": f"Sora2 style render of {os.path.basename(fpath)}", "parameters": {"num_images": 1}})
                        generated_frames.append({"source": "dalle", "frame": os.path.basename(fpath), "result": resp})
                    else:
                        generated_frames.append({"frame": os.path.basename(fpath), "note": "no image model"})
                except Exception:
                    logger.exception("per-frame generation failed", exc_info=True)
                    generated_frames.append({"frame": os.path.basename(fpath), "note": "failed"})

        # Summarize transcript
        summary = ""
        if transcript:
            try:
                local_obj, local_id = lazy_load_text_local(MODEL_REGISTRY.get("text:chat", []))
                if local_obj:
                    tok, model = local_obj
                    prompt_sum = f"Summarize the following transcript:\n\n{transcript}\n\nTl;dr:"
                    inputs = tok.encode(prompt_sum, return_tensors="pt").to(next(model.parameters()).device)
                    out = model.generate(inputs, max_new_tokens=150)
                    summary = tok.decode(out[0], skip_special_tokens=True)
                elif HF_TOKEN:
                    sm = get_best_model_id("text:summarization")
                    if sm:
                        resp = hf_inference_request(sm, transcript, params={"max_new_tokens": 150})
                        if isinstance(resp, dict) and "summary_text" in resp:
                            summary = resp["summary_text"]
                        else:
                            summary = json.dumps(resp)[:2000]
            except Exception:
                logger.exception("summarization failed", exc_info=True)
                summary = "(summarization failed)"

        return {"video_path": tmp_video, "audio_path": audio_path, "transcript": transcript, "summary": summary, "frames": frames, "generated_frames": generated_frames}
    except Exception:
        logger.exception("video pipeline failed", exc_info=True)
        raise HTTPException(500, detail="video processing failed")

# ---------------- Image worker and job queue stubs ----------------
# For production: replace with Celery/RQ + Redis + GPU workers
IMAGE_JOB_QUEUE: List[Dict[str, Any]] = []

@app.post("/worker/image/enqueue")
async def enqueue_image_job(prompt: str = Form(...), user_id: str = Form("guest")):
    job_id = uuid.uuid4().hex
    IMAGE_JOB_QUEUE.append({"job_id": job_id, "prompt": prompt, "user_id": user_id, "status": "queued", "created_at": now_ts()})
    return {"job_id": job_id}

@app.get("/worker/image/status")
async def get_image_job_status(job_id: str):
    for job in IMAGE_JOB_QUEUE:
        if job["job_id"] == job_id:
            return job
    raise HTTPException(404, detail="job not found")

# ---------------- Basic utilities endpoints ----------------
@app.get("/models")
def list_models():
    return {"models": MODEL_REGISTRY, "local_cache": list(MODEL_CACHE.keys())}

# ============================
# main.py — Zynara Ultra v5 (PART 3/3)
# Agents, memory, fusion, admin, startup, run
# ============================

# ---------------- Agents & Tool Runtime ----------------
# Lightweight agent scaffolding that composes tools (text generation, image, code, video)
AGENTS: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata

async def agent_run_loop(agent_id: str, spec: Dict[str, Any]):
    """
    Very simple autonomous agent loop:
    - spec: {"task": "...", "tools": ["generate","image","code","video"], "user_id": "..."}
    - This loop runs a fixed number of steps and uses tools via internal function calls.
    NOTE: For production use AutoGen, LangChain or custom orchestrator with safety & sandboxing.
    """
    AGENTS[agent_id]["status"] = "running"
    task_description = spec.get("task", "")
    user_id = spec.get("user_id", "guest")
    max_steps = int(spec.get("max_steps", 4))
    memory_key = f"agent:{agent_id}:history"
    history = []

    try:
        for step in range(max_steps):
            # 1) create a plan step using generate_text
            gen_req = GenerateRequest(prompt=f"Agent planning step {step+1}/{max_steps} for task: {task_description}", max_tokens=256)
            # Call internal generate_text via async call
            try:
                resp = await generate_text(gen_req, user_id=user_id)
                plan_text = resp.get("text") if isinstance(resp, dict) else str(resp)
            except Exception as e:
                plan_text = f"(planning failed: {e})"
            history.append({"step": step+1, "plan": plan_text})

            # 2) choose a tool naively by keyword
            chosen_tool = None
            plan_lower = plan_text.lower() if isinstance(plan_text, str) else ""
            if "image" in plan_lower or "draw" in plan_lower or "render" in plan_lower:
                chosen_tool = "image"
            elif "code" in plan_lower or "script" in plan_lower or "function" in plan_lower:
                chosen_tool = "code"
            elif "video" in plan_lower or "clip" in plan_lower:
                chosen_tool = "video"
            else:
                chosen_tool = "text"

            # 3) execute tool
            result = {"tool": chosen_tool, "output": None}
            try:
                if chosen_tool == "image":
                    img_req = ImageGenRequest(prompt=plan_text, width=512, height=512, samples=1)
                    img_out = await image_generate(img_req)
                    result["output"] = img_out
                elif chosen_tool == "code":
                    code_out = await run_code_in_sandbox(f'# Agent auto-generated code\n# Prompt: {plan_text}\nprint("Hello from agent step")', language="python")
                    result["output"] = code_out
                elif chosen_tool == "video":
                    # For demo: create a short stylized frame set from plan_text
                    result["output"] = {"note": "video tool executed (demo)", "plan": plan_text}
                else:
                    # text tool: one more generation
                    gen2_req = GenerateRequest(prompt=f"Agent generate answer for: {plan_text}", max_tokens=256)
                    gen2_out = await generate_text(gen2_req, user_id=user_id)
                    result["output"] = gen2_out
            except Exception as e:
                logger.exception("Agent tool call failed", exc_info=True)
                result["output"] = {"error": str(e)}

            history.append({"step_result": result})

            # 4) store intermediate memory (best-effort)
            try:
                await memory_store(user_id, f"{memory_key}:step{step+1}", {"plan": plan_text, "result": result, "ts": now_ts()})
            except Exception:
                pass

            # small delay to simulate work
            await asyncio.sleep(0.5)

        AGENTS[agent_id]["status"] = "finished"
        AGENTS[agent_id]["result"] = history
    except Exception:
        AGENTS[agent_id]["status"] = "error"
        AGENTS[agent_id]["result"] = {"error": traceback.format_exc()}

# Create agent
@app.post("/agent/create")
async def create_agent(task: str = Form(...), user_id: str = Form("guest"), max_steps: int = Form(4)):
    agent_id = uuid.uuid4().hex
    AGENTS[agent_id] = {"task": task, "user_id": user_id, "created_at": now_ts(), "status": "queued", "result": None}
    # Run in background
    spec = {"task": task, "user_id": user_id, "max_steps": max_steps}
    asyncio.create_task(agent_run_loop(agent_id, spec))
    return {"agent_id": agent_id}

@app.get("/agent/status")
def agent_status(agent_id: str):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, detail="agent not found")
    return agent

@app.get("/agent/list")
def agent_list():
    return {"agents": AGENTS}

# ---------------- Fusion Advanced Endpoint ----------------
@app.post("/fusion/advanced")
async def fusion_advanced(
    user_id: str = Form("guest"),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    voice_prompt: Optional[str] = Form(None),
    generate_images_per_video_frame: bool = Form(False),
):
    out: Dict[str, Any] = {"user_id": user_id, "results": {}}
    # process video first (if provided)
    if video:
        try:
            vres = await video_process(video)
            out["results"]["video"] = vres
        except Exception as e:
            out["results"]["video_error"] = str(e)
    # process image
    if image:
        try:
            img_bytes = await image.read()
            img_path = write_temp_file(img_bytes, ".png")
            out_img = {"path": img_path}
            # try bg removal
            try:
                if HF_TOKEN and get_best_model_id("image:inpaint"):
                    resp = await hf_inference_async(get_best_model_id("image:inpaint"), open(img_path, "rb").read())
                    out_img["bg_removed"] = True
                    out_img["bg_resp"] = resp
                else:
                    out_img["bg_removed"] = False
            except Exception:
                out_img["bg_removed"] = False
            # upscale attempt
            try:
                if HF_TOKEN and get_best_model_id("image:upscale"):
                    up = await hf_inference_async(get_best_model_id("image:upscale"), open(img_path, "rb").read())
                    out_img["upscale"] = up
            except Exception:
                out_img["upscale"] = None
            out["results"]["image"] = out_img
        except Exception as e:
            out["results"]["image_error"] = str(e)

    # text generation / summarize
    prompt_source = text or (out.get("results", {}).get("video", {}).get("transcript"))
    if prompt_source:
        try:
            ok, reason = moderate_text_basic(prompt_source)
            if not ok:
                out["results"]["text_error"] = reason
            else:
                gen_req = GenerateRequest(prompt=prompt_source, max_tokens=256)
                gen_res = await generate_text(gen_req, user_id=user_id)
                out["results"]["generated_text"] = gen_res
        except Exception as e:
            out["results"]["text_error"] = str(e)

    # voice
    if voice_prompt:
        try:
            tts_req = TTSRequest(text=voice_prompt, voice="alloy")
            # call tts endpoint internally by calling the function
            tts_resp = await tts(tts_req)
            # tts returns StreamingResponse; capture by calling TTS API directly if needed
            out["results"]["voice"] = "generated (stream)"
        except Exception as e:
            out["results"]["voice_error"] = str(e)

    # save to memory
    try:
        await memory_store(user_id, f"fusion:{stable_id(json.dumps(out.get('results') or {}))}", {"results": out.get("results"), "ts": now_ts()})
        out["memory_saved"] = True
    except Exception:
        out["memory_saved"] = False

    return out

# ---------------- Memory endpoints ----------------
@app.post("/memory/store")
async def api_memory_store(user_id: str = Form(...), key: str = Form(...), value: str = Form(...)):
    try:
        ok = await memory_store(user_id, key, {"value": value, "ts": now_ts()})
        return {"ok": ok}
    except Exception:
        logger.exception("memory store failed", exc_info=True)
        raise HTTPException(500, detail="memory store failed")

@app.get("/memory/fetch")
async def api_memory_fetch(user_id: str = Query(...), key_prefix: str = Query("")):
    try:
        mems = await memory_fetch(user_id, key_prefix)
        return {"memory": mems}
    except Exception:
        logger.exception("memory fetch failed", exc_info=True)
        raise HTTPException(500, detail="memory fetch failed")

async def memory_store(user_id: str, key: str, value: Dict[str, Any]):
    payload = {"user_id": user_id, "memkey": key, "value": json.dumps(value), "created_at": now_ts()}
    try:
        if supabase:
            supabase.table("memory").insert(payload).execute()
            return True
    except Exception:
        logger.debug("supabase memory insert failed", exc_info=True)
    try:
        if redis_client:
            redis_client.set(f"mem:{user_id}:{key}", json.dumps(value), ex=60 * 60 * 24 * 7)
            return True
    except Exception:
        logger.debug("redis memory insert failed", exc_info=True)
    return False

async def memory_fetch(user_id: str, key_prefix: str = "") -> List[Dict[str, Any]]:
    out = []
    try:
        if supabase:
            q = supabase.table("memory").select("*").eq("user_id", user_id)
            if key_prefix:
                q = q.ilike("memkey", f"{key_prefix}%")
            res = q.execute()
            if res and res.data:
                for r in res.data:
                    try:
                        rvalue = json.loads(r.get("value", "{}"))
                    except Exception:
                        rvalue = r.get("value")
                    out.append({"memkey": r.get("memkey"), "value": rvalue, "created_at": r.get("created_at")})
                return out
    except Exception:
        logger.debug("supabase fetch failed", exc_info=True)
    try:
        if redis_client:
            if key_prefix:
                keys = redis_client.keys(f"mem:{user_id}:{key_prefix}*")
            else:
                keys = redis_client.keys(f"mem:{user_id}:*")
            for k in keys:
                try:
                    kstr = k.decode() if isinstance(k, bytes) else k
                    v = redis_client.get(k)
                    if v:
                        try:
                            out.append({"memkey": kstr.split(":", 2)[-1], "value": json.loads(v)})
                        except Exception:
                            out.append({"memkey": kstr.split(":", 2)[-1], "value": v.decode() if isinstance(v, bytes) else v})
                except Exception:
                    continue
    except Exception:
        logger.debug("redis fetch failed", exc_info=True)
    return out

# ---------------- Admin / health / metrics ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf_token": bool(HF_TOKEN),
        "openai_moderation": bool(OPENAI_API_KEY),
        "eleven": bool(ELEVEN_API_KEY),
        "supabase": bool(supabase),
        "redis": bool(redis_client),
        "torch": bool(torch),
        "sdxl_local": bool(MODEL_CACHE.get("sdxl")),
        "vllm_url": VLLM_URL or None,
    }

@app.post("/admin/clear_cache")
def admin_clear():
    MODEL_CACHE.clear()
    return {"cleared": True}

@app.get("/admin/metrics")
def admin_metrics():
    # lightweight metrics
    return {
        "uptime": now_ts(),
        "agents_count": len(AGENTS),
        "image_queue_len": len(IMAGE_JOB_QUEUE),
        "mem_backend": ("supabase" if supabase else "") + (",redis" if redis_client else "")
    }

# ---------------- Startup tasks ----------------
@app.on_event("startup")
def on_startup():
    logger.info("Zynara Ultra v5 starting up")
    # warm small tokenizer if available
    try:
        if AutoTokenizer:
            AutoTokenizer.from_pretrained("google/flan-t5-small")
            logger.info("warmed small tokenizer")
    except Exception:
        logger.debug("warming tokenizer failed", exc_info=True)
    # try Milvus init
    try:
        init_milvus()
    except Exception:
        logger.debug("milvus init error", exc_info=True)
    # optionally warm local SDXL
    try:
        if USE_LOCAL_SDXL:
            lazy_load_sdxl_local()
    except Exception:
        logger.debug("sdxl warm failed", exc_info=True)
    # optionally warm whisper
    try:
        if USE_LOCAL_WHISPER:
            lazy_load_whisper_local()
    except Exception:
        logger.debug("whisper warm failed", exc_info=True)

# ---------------- Run server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
