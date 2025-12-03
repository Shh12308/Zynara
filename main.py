# main.py — Zynara Super-Ultimate: full multimodal backend
# WARNING: This is a large, feature-rich demo. Configure tokens, GPU, and infra before running.

import os
import io
import sys
import time
import json
import uuid
import math
import shutil
import logging
import asyncio
import traceback
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional libs imported lazily
try:
    import torch
except Exception:
    torch = None

try:
    import httpx
except Exception:
    httpx = None

# transformers / diffusers / whisper / PIL - used if available locally
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, AutoModel
except Exception:
    AutoTokenizer = AutoModelForCausalLM = pipeline = AutoModelForSeq2SeqLM = AutoModel = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# Supabase and Redis clients (optional)
try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

try:
    import redis as redis_lib
except Exception:
    redis_lib = None

# Optional: vLLM/text-generation-inference integration would go here for best performance.

# --------------------------- Logging ---------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zynara")

# --------------------------- Config ---------------------------
APP_NAME = os.getenv("APP_NAME", "Zynara Super-Ultimate")
CREATOR = os.getenv("APP_AUTHOR", "GoldBoy")
PORT = int(os.getenv("PORT", "7860"))
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face Inference token
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional moderation / fallback
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")  # ElevenLabs TTS
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_URL = os.getenv("REDIS_URL")
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "1") == "1"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

TMP_DIR = os.getenv("TMP_DIR", "/tmp/zynara")
os.makedirs(TMP_DIR, exist_ok=True)
MEDIA_DIR = os.path.join(TMP_DIR, "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# --------------------------- App ---------------------------
app = FastAPI(title=APP_NAME, description=f"{APP_NAME} by {CREATOR}")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------- Optional Clients ---------------------------
supabase = None
if create_supabase_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase initialized")
    except Exception as e:
        logger.warning("Supabase init failed: %s", e)

redis_client = None
if redis_lib and REDIS_URL:
    try:
        redis_client = redis_lib.from_url(REDIS_URL)
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis init failed: %s", e)

# --------------------------- Utilities ---------------------------
def now_ts() -> int:
    return int(time.time())

def write_temp_file(data: bytes, suffix: str = "") -> str:
    path = os.path.join(MEDIA_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path

def stable_id(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def cleanup_temp_files(older_than_seconds: int = 3600):
    now = time.time()
    for fn in os.listdir(MEDIA_DIR):
        path = os.path.join(MEDIA_DIR, fn)
        try:
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > older_than_seconds:
                os.remove(path)
        except Exception:
            pass

# --------------------------- Moderation ---------------------------
def moderate_text_basic(text: str) -> Tuple[bool, Optional[str]]:
    if not text:
        return True, None
    banned = ["bomb", "explode", "kill", "terror", "suicide"]
    for b in banned:
        if b in text.lower():
            return False, f"Blocked word: {b}"
    # Optionally call OpenAI moderation if key present
    if OPENAI_API_KEY and httpx:
        try:
            r = httpx.post("https://api.openai.com/v1/moderations", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, json={"input": text}, timeout=10.0)
            if r.status_code == 200:
                jr = r.json()
                if jr.get("results") and jr["results"][0].get("flagged"):
                    return False, "OpenAI moderation flagged content"
        except Exception:
            logger.debug("OpenAI moderation failed", exc_info=True)
    return True, None

# --------------------------- MODEL REGISTRY ---------------------------
MODEL_REGISTRY: Dict[str, List[str]] = {
    "text:chat":["meta-llama/Llama-3-70B","tiiuae/falcon-180b"],
    "code:gen":["bigcode/starcoder","Salesforce/codegen-6B-multi"],
    "image:sdxl":["stabilityai/stable-diffusion-xl-base-1.0"],
    "vision:caption":["Salesforce/blip2-flan-t5-xl"],
    "vision:vqa":["Salesforce/blip-vqa-large"],
    "speech:stt":["openai/whisper-large-v2"],
    "audio:music":["facebook/musicgen-large"],
    "vision:detector":["ultralytics/yolov8x"],
    "vision:ocr":["microsoft/trocr-large-handwritten"],
    "image:inpaint":["stabilityai/stable-diffusion-x4-inpainting"],
    "image:upscale":["nateraw/real-esrgan"],
    "text:embed":["sentence-transformers/all-mpnet-base-v2"],
    "text:summarization":["facebook/bart-large-cnn"],
    "text:translation":["Helsinki-NLP/opus-mt-en-ROMANCE"],
    "text:sentiment":["distilbert-base-uncased-finetuned-sst-2-english"],
    "document:layoutlm":["layoutlmv3-base"],
    "3d:pointcloud":["openai/point-e"],
    "ml:anomaly":["anomaly-detection-model"],
    "ml:game_ai":["game-behavior-model"],
    "image:style_transfer":["style-transfer-model"],
    "vision:pose2anim":["pose-to-character-model"]
}

MODEL_CACHE: Dict[str, Any] = {}  # key -> loaded pipeline/model object

def get_best_model_id(category_key: str) -> Optional[str]:
    lst = MODEL_REGISTRY.get(category_key, [])
    return lst[0] if lst else None

# --------------------------- Lazy loaders ---------------------------
def lazy_load_text_local(preferred: List[str]):
    """
    Try to load a text model locally. Skips extremely large models if environment
    doesn't have GPU or transformers support. Returns (tokenizer, model), model_id.
    """
    key = "text:local"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key], MODEL_CACHE.get(f"{key}:id")
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, preferred[0] if preferred else None
    for mid in preferred:
        # Avoid loading absurdly large models by default; let user control via env
        try:
            logger.info("Attempting to load local text model %s", mid)
            tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")
            MODEL_CACHE[key] = (tok, model)
            MODEL_CACHE[f"{key}:id"] = mid
            return (tok, model), mid
        except Exception as e:
            logger.warning("Local text load failed for %s: %s", mid, e)
            continue
    return None, preferred[0] if preferred else None

def lazy_load_sdxl_local():
    key = "sdxl"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    if StableDiffusionPipeline is None or torch is None:
        logger.info("StableDiffusionPipeline or torch not available locally")
        return None
    model_id = get_best_model_id("image:sdxl")
    if not model_id:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SDXL local pipeline %s on %s", model_id, device)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        if device == "cuda":
            pipe = pipe.to("cuda")
        MODEL_CACHE[key] = pipe
        return pipe
    except Exception as e:
        logger.exception("Failed to load SDXL locally: %s", e)
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
        return w
    except Exception as e:
        logger.exception("Failed to load Whisper locally: %s", e)
        return None

# --------------------------- Hugging Face Inference wrappers ---------------------------
def hf_inference_request(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    if httpx is None:
        raise RuntimeError("httpx not available")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        if isinstance(inputs, (bytes, bytearray)):
            r = httpx.post(url, headers=headers, content=inputs, timeout=timeout)
        else:
            body = {"inputs": inputs}
            if params:
                body["parameters"] = params
            r = httpx.post(url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception("HF inference request failed for %s: %s", model_id, e)
        raise

async def hf_inference_async(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    if httpx is None:
        raise RuntimeError("httpx not available")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
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
    except Exception as e:
        logger.exception("HF async inference failed: %s", e)
        raise

# --------------------------- Pydantic Request models ---------------------------
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

ELEVEN_VOICES = ["alloy","bella","dennis","clara","mike","emma","sophia"]

# --------------------------- Adaptive memory (Supabase/Redis) ---------------------------
async def memory_store(user_id: str, key: str, value: Dict[str, Any]):
    payload = {"user_id": user_id, "memkey": key, "value": json.dumps(value), "created_at": now_ts()}
    try:
        if supabase:
            supabase.table("memory").insert(payload).execute()
            return True
    except Exception as e:
        logger.warning("supabase memory insert failed: %s", e)
    try:
        if redis_client:
            redis_client.set(f"mem:{user_id}:{key}", json.dumps(value), ex=60*60*24*7)
            return True
    except Exception:
        pass
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
    except Exception as e:
        logger.debug("supabase memory fetch failed: %s", e)
    # Redis fallback
    try:
        if redis_client and key_prefix:
            keys = redis_client.keys(f"mem:{user_id}:{key_prefix}*")
            for k in keys:
                v = redis_client.get(k)
                if v:
                    try:
                        out.append({"memkey": k.decode().split(":",2)[-1], "value": json.loads(v)})
                    except Exception:
                        out.append({"memkey": k.decode().split(":",2)[-1], "value": v.decode()})
    except Exception:
        pass
    return out

# --------------------------- Rate limiting ---------------------------
RATE_LIMIT = {}  # user_id -> (tokens, last_ts)
RATE_TOKENS = 60
RATE_WINDOW = 60

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

# --------------------------- Endpoints ---------------------------

@app.get("/")
def index():
    return {"app": APP_NAME, "creator": CREATOR, "note": "Zynara Super-Ultimate — multimodal AI backend. Configure HF_TOKEN/OPENAI/ELEVEN/SUPABASE/REDIS."}

# --------------------------- Health & Admin ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf_token": bool(HF_TOKEN),
        "openai_key": bool(OPENAI_API_KEY),
        "eleven": bool(ELEVEN_API_KEY),
        "supabase": bool(supabase),
        "redis": bool(redis_client),
        "torch": bool(torch),
        "sdxl_local": bool(MODEL_CACHE.get("sdxl")),
        "text_local": bool(MODEL_CACHE.get("text:local"))
    }

@app.post("/admin/clear_cache")
def admin_clear():
    MODEL_CACHE.clear()
    return {"cleared": True}

@app.get("/admin/models")
def admin_models():
    return {"registry": MODEL_REGISTRY, "loaded": list(MODEL_CACHE.keys())}

# --------------------------- Text generation & chat ---------------------------
@app.post("/generate")
def generate_text(req: GenerateRequest, user_id: str = Form("guest")):
    ok, reason = moderate_text_basic(req.prompt)
    if not ok:
        raise HTTPException(400, detail=reason or "Blocked")
    if not consume_token(user_id, cost=1):
        raise HTTPException(429, detail="Rate limit exceeded")
    preferred = MODEL_REGISTRY.get(req.model_hint or "text:chat", MODEL_REGISTRY.get("text:chat", []))
    local_obj, local_id = lazy_load_text_local(preferred)
    if local_obj:
        try:
            tok, model = local_obj
            inputs = tok.encode(req.prompt, return_tensors="pt").to(next(model.parameters()).device)
            out = model.generate(inputs, max_new_tokens=req.max_tokens)
            text = tok.decode(out[0], skip_special_tokens=True)
            return {"source": "local", "model_id": local_id, "text": text}
        except Exception:
            logger.exception("Local text generation failed")
    # HF inference fallback
    if HF_TOKEN and USE_HF_INFERENCE:
        model_id = local_id or get_best_model_id(req.model_hint or "text:chat")
        try:
            resp = hf_inference_request(model_id, req.prompt, params={"max_new_tokens": req.max_tokens, "temperature": req.temperature})
            # extract text safely
            if isinstance(resp, dict) and "generated_text" in resp:
                out_text = resp["generated_text"]
            elif isinstance(resp, list) and isinstance(resp[0], dict) and "generated_text" in resp[0]:
                out_text = resp[0]["generated_text"]
            else:
                out_text = json.dumps(resp)[:8000]
            return {"source": "hf", "model_id": model_id, "text": out_text}
        except Exception as e:
            logger.exception("HF generation failed: %s", e)
            raise HTTPException(status_code=500, detail="HF generation failed")
    raise HTTPException(status_code=503, detail="No text model available")

# --------------------------- WebSocket streaming (chat) ---------------------------
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        init = await websocket.receive_json()
        prompt = init.get("prompt", "")
        user_id = init.get("user_id", "guest")
        ok, reason = moderate_text_basic(prompt)
        if not ok:
            await websocket.send_json({"error": reason})
            await websocket.close()
            return
        if not consume_token(user_id, cost=1):
            await websocket.send_json({"error":"rate_limit"})
            await websocket.close()
            return
        # Try local streaming (naive chunked approach)
        preferred = MODEL_REGISTRY.get("text:chat", [])
        local_obj, local_id = lazy_load_text_local(preferred)
        if local_obj:
            try:
                tok, model = local_obj
                inputs = tok.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
                out = model.generate(inputs, max_new_tokens=1024)
                text = tok.decode(out[0], skip_special_tokens=True)
                for i in range(0, len(text), 120):
                    await websocket.send_json({"delta": text[i:i+120]})
                    await asyncio.sleep(0.02)
                await websocket.send_json({"done": True})
                await websocket.close()
                return
            except Exception:
                logger.exception("Local ws streaming failed")
        # HF fallback — simulate streaming by chunking final text
        model_id = get_best_model_id("text:chat")
        try:
            resp = hf_inference_request(model_id, prompt, params={"max_new_tokens": 1024})
            if isinstance(resp, dict) and "generated_text" in resp:
                text = resp["generated_text"]
            elif isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
                text = resp[0]["generated_text"]
            else:
                text = json.dumps(resp)[:8000]
            for i in range(0, len(text), 120):
                await websocket.send_json({"delta": text[i:i+120]})
                await asyncio.sleep(0.02)
            await websocket.send_json({"done": True})
            await websocket.close()
            return
        except Exception:
            await websocket.send_json({"error": "generation_failed"})
            await websocket.close()
            return
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("ws_chat error")
        try:
            await websocket.close()
        except Exception:
            pass

# --------------------------- Code generation endpoint ---------------------------
@app.post("/code")
def code_generate(prompt: str = Form(...), language: str = Form("python"), model_hint: str = Form("code:gen"), user_id: str = Form("guest")):
    ok, reason = moderate_text_basic(prompt)
    if not ok:
        raise HTTPException(400, detail=reason)
    if not consume_token(user_id, cost=2):
        raise HTTPException(429, detail="Rate limit exceeded")
    preferred = MODEL_REGISTRY.get(model_hint, MODEL_REGISTRY.get("code:gen", []))
    local_obj, local_id = lazy_load_text_local(preferred)
    if local_obj:
        try:
            tok, model = local_obj
            composed = f"# Language: {language}\n# Request: {prompt}\n\n"
            inputs = tok.encode(composed, return_tensors="pt").to(next(model.parameters()).device)
            out = model.generate(inputs, max_new_tokens=1024)
            code_text = tok.decode(out[0], skip_special_tokens=True)
            return {"source": "local", "model": local_id, "code": code_text}
        except Exception:
            logger.exception("Local code generation failed")
    # HF fallback
    if HF_TOKEN and USE_HF_INFERENCE:
        model_id = local_id or get_best_model_id(model_hint)
        try:
            resp = hf_inference_request(model_id, f"Write a {language} program: {prompt}", params={"max_new_tokens": 1024})
            if isinstance(resp, dict) and "generated_text" in resp:
                return {"source": "hf", "model": model_id, "code": resp["generated_text"]}
            return {"source": "hf", "model": model_id, "code_raw": resp}
        except Exception:
            logger.exception("HF code gen failed")
    raise HTTPException(status_code=503, detail="No code model available")

# --------------------------- Image generation (SDXL) ---------------------------
@app.post("/image/generate")
async def image_generate(req: ImageGenRequest, user_id: str = Form("guest")):
    ok, reason = moderate_text_basic(req.prompt)
    if not ok:
        raise HTTPException(400, detail=reason)
    if not consume_token(user_id, cost=3):
        raise HTTPException(429, detail="Rate limit exceeded")
    # Try local SDXL
    sd = lazy_load_sdxl_local()
    if sd:
        try:
            out = sd(req.prompt, num_inference_steps=30, width=req.width, height=req.height, guidance_scale=7.5)
            img = out.images[0]
            out_path = os.path.join(MEDIA_DIR, f"img_{uuid.uuid4().hex}.png")
            img.save(out_path)
            return FileResponse(out_path, media_type="image/png", filename=os.path.basename(out_path))
        except Exception:
            logger.exception("Local SDXL failed")
    # HF inference fallback
    if HF_TOKEN and USE_HF_INFERENCE:
        model_id = get_best_model_id(req.model_hint or "image:sdxl")
        try:
            resp = await hf_inference_async(model_id, {"prompt": req.prompt, "width": req.width, "height": req.height}, timeout=180)
            # handle common HF return shapes (base64 or url)
            if isinstance(resp, dict) and "images" in resp:
                b64 = resp["images"][0]
                import base64
                img_bytes = base64.b64decode(b64)
                path = write_temp_file(img_bytes, ".png")
                return FileResponse(path, media_type="image/png", filename=os.path.basename(path))
            return JSONResponse({"source": "hf", "result": resp})
        except Exception:
            logger.exception("HF image gen failed")
    raise HTTPException(status_code=503, detail="No image generator available")

# --------------------------- Image inpaint / remove_bg / upscale / style transfer ---------------------------
@app.post("/image/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form(""), user_id: str = Form("guest")):
    tmp_image = write_temp_file(await image.read(), os.path.splitext(image.filename)[1] if image.filename else ".png")
    tmp_mask = write_temp_file(await mask.read(), os.path.splitext(mask.filename)[1] if mask.filename else ".png")
    model_id = get_best_model_id("image:inpaint")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"image": open(tmp_image, "rb").read(), "mask": open(tmp_mask, "rb").read(), "prompt": prompt})
            return JSONResponse({"source": "hf", "result": resp})
        except Exception:
            logger.exception("HF inpaint failed")
    raise HTTPException(status_code=503, detail="Inpaint unavailable")

@app.post("/image/upscale")
async def upscale(image: UploadFile = File(...), scale: int = Form(2), user_id: str = Form("guest")):
    tmp_image = write_temp_file(await image.read(), os.path.splitext(image.filename)[1] if image.filename else ".png")
    model_id = get_best_model_id("image:upscale")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp_image, "rb").read(), params={"scale": scale})
            return JSONResponse({"source":"hf","result":resp})
        except Exception:
            logger.exception("HF upscale failed")
    raise HTTPException(status_code=503, detail="Upscaler unavailable")

@app.post("/image/style_transfer")
async def style_transfer(image: UploadFile = File(...), style: str = Form(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".png")
    model_id = get_best_model_id("image:style_transfer")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"image": open(tmp,"rb").read(), "style": style})
            return JSONResponse({"source":"hf","result":resp})
        except Exception:
            logger.exception("style transfer failed")
    raise HTTPException(status_code=503, detail="Style transfer unavailable")

# --------------------------- Vision: caption / vqa / detect / ocr / pose --------------------------------
@app.post("/vision/caption")
async def caption(image: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".png")
    model_id = get_best_model_id("vision:caption")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return JSONResponse({"source":"hf","caption":resp})
        except Exception:
            logger.exception("HF caption failed")
    raise HTTPException(status_code=503, detail="Caption model unavailable")

@app.post("/vision/vqa")
async def vqa(image: UploadFile = File(...), question: str = Form(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".png")
    model_id = get_best_model_id("vision:vqa")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"image": open(tmp,"rb").read(), "question": question})
            return JSONResponse({"source":"hf","answer":resp})
        except Exception:
            logger.exception("HF vqa failed")
    raise HTTPException(status_code=503, detail="VQA model unavailable")

@app.post("/vision/detect")
async def detect(image: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".jpg")
    model_id = get_best_model_id("vision:detector")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return JSONResponse({"source":"hf","result":resp})
        except Exception:
            logger.exception("HF detection failed")
    raise HTTPException(status_code=503, detail="Object detection unavailable")

@app.post("/vision/ocr")
async def ocr(image: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".png")
    model_id = get_best_model_id("vision:ocr")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return JSONResponse({"source":"hf","text":resp})
        except Exception:
            logger.exception("HF ocr failed")
    raise HTTPException(status_code=503, detail="OCR unavailable")

@app.post("/vision/pose2anim")
async def pose2anim(image: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await image.read(), ".png")
    model_id = get_best_model_id("vision:pose2anim")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return JSONResponse({"source":"hf","animation":resp})
        except Exception:
            logger.exception("HF pose2anim failed")
    raise HTTPException(status_code=503, detail="Pose-to-animation unavailable")

# --------------------------- Document parsing / Layout / PDF --------------------------------
@app.post("/document/parse")
async def document_parse(file: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await file.read(), ".pdf")
    model_id = get_best_model_id("document:layoutlm")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return JSONResponse({"source":"hf","parsed":resp})
        except Exception:
            logger.exception("Document parse failed")
    raise HTTPException(status_code=503, detail="Document parse unavailable")

# --------------------------- Speech: STT & TTS & voice cloning ---------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await file.read(), os.path.splitext(file.filename)[1] if file.filename else ".wav")
    whisper = lazy_load_whisper_local()
    if whisper:
        try:
            segments, info = whisper.transcribe(tmp)
            text = " ".join([s.text for s in segments])
            return {"source":"local","text":text}
        except Exception:
            logger.exception("Local whisper failed")
    model_id = get_best_model_id("speech:stt")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, open(tmp,"rb").read())
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("HF stt failed")
    raise HTTPException(status_code=503, detail="STT unavailable")

@app.post("/tts")
async def tts(req: TTSRequest, user_id: str = Form("guest")):
    ok, reason = moderate_text_basic(req.text)
    if not ok:
        raise HTTPException(400, detail=reason or "Blocked")
    # Prefer ElevenLabs if configured
    if ELEVEN_API_KEY and httpx:
        try:
            voice = req.voice or "alloy"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
            headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
            payload = {"text": req.text}
            r = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            if r.status_code == 200:
                path = write_temp_file(r.content, ".mp3")
                return FileResponse(path, media_type="audio/mpeg")
            else:
                logger.warning("ElevenLabs returned %s %s", r.status_code, r.text[:200])
        except Exception:
            logger.exception("ElevenLabs TTS failed")
    # HF / local fallback - try any TTS model via HF
    model_id = get_best_model_id("speech:tts") or get_best_model_id("speech:stt")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, req.text)
            return JSONResponse({"source":"hf","result":resp})
        except Exception:
            logger.exception("HF tts failed")
    raise HTTPException(status_code=503, detail="TTS unavailable")

@app.post("/voice_clone")
async def voice_clone(seed_audio: UploadFile = File(...), text: str = Form(...), user_id: str = Form("guest")):
    # Placeholder: ElevenLabs voice cloning flow or 3rd party voice cloning would go here.
    if ELEVEN_API_KEY:
        # NOTE: voice cloning via Eleven Labs requires a specific workflow & account access.
        return JSONResponse({"note": "Voice cloning with ElevenLabs - implement API call here (requires voice cloning enabled)."})
    return JSONResponse({"error": "Voice cloning requires ElevenLabs or another provider key."}, status_code=503)

# --------------------------- Music generation & editing ---------------------------
@app.post("/music/generate")
async def music_generate(prompt: str = Form(...), duration: int = Form(20), user_id: str = Form("guest")):
    model_id = get_best_model_id("audio:music")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt, "duration": duration})
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("HF musicgen failed")
    raise HTTPException(status_code=503, detail="Music generation unavailable")

@app.post("/music/edit")
async def music_edit(track: UploadFile = File(...), style: str = Form(...), user_id: str = Form("guest")):
    tmp = write_temp_file(await track.read(), ".mp3")
    model_id = get_best_model_id("audio:music")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"track": open(tmp,"rb").read(), "style": style})
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("HF music edit failed")
    raise HTTPException(status_code=503, detail="Music editing unavailable")

# --------------------------- Video: text2video / video summary / img2vid placeholders ---------------------------
@app.post("/video/text2video")
async def text2video(prompt: str = Form(...), seconds: int = Form(4), user_id: str = Form("guest")):
    model_id = get_best_model_id("video:text2video")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt, "seconds": seconds})
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("HF text2video failed")
    return JSONResponse({"error": "Text2Video requires heavy models; configure HF_TOKEN or local pipeline."}, status_code=503)

@app.post("/fusion/video_summary")
async def fusion_video_summary(video: UploadFile = File(...), user_id: str = Form("guest")):
    # Save and queue heavy processing in production. Here: placeholder pipeline.
    vbytes = await video.read()
    vpath = write_temp_file(vbytes, os.path.splitext(video.filename)[1] if video.filename else ".mp4")
    # Placeholder: use ffmpeg to extract audio, then whisper to transcribe, then summarise.
    transcript = "(video transcription placeholder — implement ffmpeg + whisper or HF speech-to-text)"
    model_id = get_best_model_id("text:summarization")
    if HF_TOKEN and model_id:
        try:
            summary = hf_inference_request(model_id, transcript, params={"max_new_tokens": 200})
            return {"transcript": transcript, "summary": summary}
        except Exception:
            logger.exception("Video summary failed")
            return {"transcript": transcript, "summary": "(summary failed)"}
    return {"transcript": transcript, "summary": "(no summarizer configured)"}

# --------------------------- 3D: pointcloud / mesh placeholders ---------------------------
@app.post("/3d/generate")
async def generate_pointcloud(prompt: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("3d:pointcloud")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt})
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("3D generation failed")
    return JSONResponse({"error":"3D generation requires Point-E or similar model"}, status_code=503)

# --------------------------- Embeddings & RAG helpers ---------------------------
@app.post("/text/embed")
def embed_text(texts: List[str] = Form(...), user_id: str = Form("guest")):
    try:
        from sentence_transformers import SentenceTransformer
        model = MODEL_CACHE.get("embed")
        if not model:
            model_name = get_best_model_id("text:embed") or "sentence-transformers/all-mpnet-base-v2"
            model = SentenceTransformer(model_name)
            MODEL_CACHE["embed"] = model
        vecs = model.encode(texts).tolist()
        return {"source":"local","embeddings":vecs}
    except Exception:
        pass
    if HF_TOKEN and USE_HF_INFERENCE:
        model_id = get_best_model_id("text:embed") or "sentence-transformers/all-mpnet-base-v2"
        try:
            resp = hf_inference_request(model_id, {"inputs": texts})
            return {"source":"hf","embeddings":resp}
        except Exception:
            logger.exception("HF embed failed")
    raise HTTPException(status_code=503, detail="Embedding model unavailable")

# --------------------------- Text analysis: summarise / translate / sentiment / finance / medical ---------------------------
@app.post("/text/summarize")
def text_summarize(text: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("text:summarization")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, text)
            return {"source":"hf","summary":resp}
        except Exception:
            logger.exception("Summarization failed")
    raise HTTPException(status_code=503, detail="Summarization unavailable")

@app.post("/text/translate")
def translate(text: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("text:translation")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, text)
            return {"source":"hf","translation":resp}
        except Exception:
            logger.exception("Translation failed")
    raise HTTPException(status_code=503, detail="Translation unavailable")

@app.post("/analyze/sentiment")
def sentiment_analysis(text: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("text:sentiment")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, text)
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("Sentiment analysis failed")
    raise HTTPException(status_code=503, detail="Sentiment model unavailable")

@app.post("/analyze/finance")
def finance_analysis(text: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("text:finance")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, text)
            return {"source":"hf","analysis":resp}
        except Exception:
            logger.exception("Finance analysis failed")
    raise HTTPException(status_code=503, detail="Finance analysis model unavailable")

@app.post("/analyze/medical")
def medical_analysis(text: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("text:medical")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, text)
            return {"source":"hf","analysis":resp}
        except Exception:
            logger.exception("Medical analysis failed")
    raise HTTPException(status_code=503, detail="Medical analysis model unavailable")

# --------------------------- ML endpoints: anomaly / game ai / recommend ---------------------------
@app.post("/analyze/anomaly")
async def anomaly_detection(data: List[float] = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("ml:anomaly")
    if HF_TOKEN and model_id:
        try:
            resp = await hf_inference_async(model_id, {"input": data})
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("Anomaly detection failed")
    return {"note":"Local anomaly detection not implemented"}

@app.post("/ml/game_ai")
async def game_ai_simulation(context: str = Form(...), user_id: str = Form("guest")):
    model_id = get_best_model_id("ml:game_ai")
    if HF_TOKEN and model_id:
        try:
            resp = hf_inference_request(model_id, context)
            return {"source":"hf","result":resp}
        except Exception:
            logger.exception("Game AI failed")
    raise HTTPException(status_code=503, detail="Game AI unavailable")

# --------------------------- Adaptive memory endpoints ---------------------------
@app.post("/memory/store")
async def api_memory_store(user_id: str = Form(...), key: str = Form(...), value: str = Form(...)):
    try:
        ok = await memory_store(user_id, key, {"value": value, "ts": now_ts()})
        return {"ok": ok}
    except Exception:
        logger.exception("memory store failed")
        raise HTTPException(500, detail="memory store failed")

@app.get("/memory/fetch")
async def api_memory_fetch(user_id: str = Query(...), key_prefix: str = Query("")):
    try:
        mems = await memory_fetch(user_id, key_prefix)
        return {"memory": mems}
    except Exception:
        logger.exception("memory fetch failed")
        raise HTTPException(500, detail="memory fetch failed")

# --------------------------- WebSocket STT (streaming audio) ---------------------------
@app.websocket("/ws/stt")
async def ws_stt(websocket: WebSocket):
    await websocket.accept()
    try:
        init = await websocket.receive_json()
        user_id = init.get("user_id", "guest")
        await websocket.send_json({"status":"ready"})
        collected = bytearray()
        while True:
            msg = await websocket.receive_json()
            if msg.get("done"):
                break
            b64 = msg.get("chunk")
            if not b64:
                continue
            import base64
            collected.extend(base64.b64decode(b64))
            # optional partial transcription placeholder
            if len(collected) > 16000*10:
                await websocket.send_json({"partial":"(partial transcript placeholder)"})
        tmp = write_temp_file(bytes(collected), ".wav")
        whisper = lazy_load_whisper_local()
        if whisper:
            try:
                segments, info = whisper.transcribe(tmp)
                text = " ".join([s.text for s in segments])
                await websocket.send_json({"result": text})
                await websocket.close()
                return
            except Exception:
                logger.exception("Local whisper transcribe failed")
        # HF fallback
        await websocket.send_json({"result":"(transcription placeholder - enable whisper or HF stt)"})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("stt websocket disconnected")
    except Exception:
        logger.exception("ws_stt error")
        try:
            await websocket.close()
        except:
            pass

# --------------------------- Misc helpers ---------------------------
@app.get("/models/registry")
def models_registry():
    return {"registry": MODEL_REGISTRY}

@app.get("/models/find")
def models_find(category: str = Query(...)):
    return {"category": category, "candidates": MODEL_REGISTRY.get(category, [])}

@app.post("/admin/cleanup_temp")
def admin_cleanup_temp(older_than_seconds: int = Form(3600)):
    cleanup_temp_files(older_than_seconds)
    return {"ok": True}

# --------------------------- Startup tasks ---------------------------
@app.on_event("startup")
def on_startup():
    logger.info("Starting Zynara Super-Ultimate")
    logger.info("HF_TOKEN set: %s", bool(HF_TOKEN))
    logger.info("OpenAI key set: %s", bool(OPENAI_API_KEY))
    logger.info("ElevenLabs key set: %s", bool(ELEVEN_API_KEY))
    logger.info("Torch available: %s", bool(torch))
    # Warm small tokenizer to reduce first-request latency
    if AutoTokenizer:
        try:
            AutoTokenizer.from_pretrained("google/flan-t5-small")
            logger.info("Warmed small tokenizer")
        except Exception:
            pass

# --------------------------- Run if main ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
