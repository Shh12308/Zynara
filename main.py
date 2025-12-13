# main.py — Zynara Mega AI Backend (1000+ lines)
# Multi-modal AI backend integrating 50+ HF models, local + streaming + caching

import os
import io
import time
import uuid
import json
import hashlib
import traceback
from typing import Optional, List, Dict, Any, Tuple
from threading import Thread

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, WebSocket, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional heavy imports
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        pipeline
    )
except Exception:
    AutoTokenizer = AutoModelForCausalLM = AutoModelForSeq2SeqLM = pipeline = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

try:
    import redis as redis_lib
except Exception:
    redis_lib = None

try:
    from PIL import Image
except Exception:
    Image = None

# ===============================
# Multimodal libs
# ===============================
try:
    from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
except Exception:
    StableDiffusionPipeline = StableVideoDiffusionPipeline = None

try:
    # LTX‑2 video generation (if installed)
    from ltx2 import LTX2Pipeline
except Exception:
    LTX2Pipeline = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None

# ===============================
# App config
# ===============================
APP_NAME = os.getenv("APP_NAME", "Zynara Mega AI")
CREATOR = os.getenv("APP_AUTHOR", "GoldYLocks")
PORT = int(os.getenv("PORT", 7860))
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Multi-modal AI backend built and designed using hf models")

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_URL = os.getenv("REDIS_URL")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
WOLFRAM_KEY = os.getenv("WOLFRAM_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
DISABLE_MULTIMODAL = os.getenv("DISABLE_MULTIMODAL", "0") == "1"
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "1") == "1"

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

app = FastAPI(title=APP_NAME, description=APP_DESCRIPTION)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"], allow_methods=["*"], allow_headers=["*"])

# ===============================
# Clients
# ===============================
supabase = None
if create_supabase_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_supabase_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)

redis_client = None
if redis_lib and REDIS_URL:
    try:
        redis_client = redis_lib.from_url(REDIS_URL)
        print("✅ Redis connected")
    except Exception as e:
        print("⚠️ Redis init failed:", e)

OPENAI_MOD=None
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        OPENAI_MOD = openai
        print("✅ OpenAI client available for moderation")
    except Exception:
        OPENAI_MOD = None

# ===============================
# Utilities
# ===============================
MODEL_CACHE: Dict[str, Any] = {}

def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def cache_get(key: str):
    if not redis_client:
        return None
    try:
        v = redis_client.get(key)
        if not v:
            return None
        return json.loads(v)
    except Exception:
        return None

def cache_set(key: str, value, ttl: int = 300):
    if not redis_client:
        return
    try:
        redis_client.set(key, json.dumps(value), ex=ttl)
    except Exception:
        pass

def moderate_text(text: str) -> Tuple[bool, Optional[str]]:
    if not text:
        return True, None
    if OPENAI_MOD:
        try:
            resp = OPENAI_MOD.Moderation.create(input=text)
            flagged = any(resp["results"][0]["categories"].values()) or resp["results"][0].get("flagged", False)
            return (not flagged, "OpenAI moderation blocked" if flagged else None)
        except Exception:
            pass
    banned = ["bomb", "kill", "terror", "explosive"]
    if any(b in text.lower() for b in banned):
        return False, "Blocked by heuristic"
    return True, None

def get_best_hf_id(category_key: str) -> Optional[str]:
    lst = MODEL_REGISTRY.get(category_key, [])
    return lst[0] if lst else None

# ===============================
# Model Registry — 50+ HF models across 10 categories
# ===============================
MODEL_REGISTRY = {
    # Text / NLP
    "text:chat": ["meta-llama/Llama-2-70b-chat", "tiiuae/falcon-180B"],
    "text:instruct": ["google/flan-ul2", "t5-3b"],
    "text:summarize": ["facebook/bart-large-cnn", "google/pegasus-large"],
    "text:qa": ["deepset/roberta-base-squad2", "valhalla/distilbart-mnli-12-6"],
    "text:translate": ["facebook/mbart-large-50", "Helsinki-NLP/opus-mt-en-fr"],
    "text:sentiment": ["cardiffnlp/twitter-roberta-base-sentiment-latest"],
    "text:embed": ["sentence-transformers/all-mpnet-base-v2", "all-MiniLM-L6-v2"],
    "text:ner": ["dbmdz/bert-large-cased-finetuned-conll03-english"],
    "text:moderation": ["unitary/toxic-bert"],

    # Code / Programming
    "code:gen": ["bigcode/starcoder", "Salesforce/codegen-6B-multi"],
    "code:assist": ["codellama/CodeLlama-7b-instruct"],
    "code:summarize": ["Salesforce/codet5-large-multi-sum"],
    "code:embed": ["microsoft/codebert-base"],

    # Vision / Image
    "vision:classify": ["google/vit-base-patch16-224"],
    "vision:detector": ["facebook/detr-resnet-101"],
    "vision:segment": ["facebook/segformer-b5-finetuned-ade-512-512"],
    "vision:pose": ["facebook/detectron2"],
    "image:sdxl": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "image:inpaint": ["stabilityai/stable-diffusion-x4-inpainting"],
    "image:upscale": ["nateraw/real-esrgan"],
    "image:bg_remove": ["photoroom/background-removal"],
    "image:style_transfer": ["CompVis/stable-diffusion-v1-4"],
    "vision:ocr": ["microsoft/trocr-large-handwritten"],

    # Audio / Speech
    "speech:tts": ["tts_models/en/vctk/vits"],
    "speech:whisper": ["openai/whisper-large-v2"],
    "speech:voice_clone": ["facebook/yourtts"],
    "speech:enhance": ["facebook/segan"],
    "audio:musicgen": ["facebook/musicgen-large"],

    # Multimodal
    "vision:vqa": ["Salesforce/blip-vqa-large"],
    "vision:caption": ["Salesforce/blip2-flan-t5-xl"],
    "image:txt2img": ["stabilityai/stable-diffusion-xl-base-1.0"],
"video:txt2vid": ["lightricks/ltx2-text2video"],

    # Video
    "video:classify": ["facebook/timesformer-base-finetuned-k400"],
    "video:img2vid": ["stabilityai/stable-video-diffusion"],

    # 3D / Geometry
    "3d:object": ["openai/point-e"],
    "3d:nerf": ["nerfstudio/nerfacto"],
    "3d:mesh": ["facebookresearch/mesh-rcnn"],

    # RL / Control
    "rl:policy": ["stable-baselines3/ppo"],
    "rl:decision": ["DecisionTransformer"],

    # ML Utilities
    "ml:anomaly": ["openai/clip-vit-base-patch32"],
    "ml:recommender": ["microsoft/recommenders"],
    "ml:feature": ["openai/clip-vit-large-patch14"],

    # Medical / Scientific
    "medical:imaging": ["stanfordmlgroup/chexpert"],
    "medical:protein": ["facebook/esm2_t36_3B_UR50D"],
}

# ===============================
# Lazy-loading helpers
# ===============================
def load_text_model(category_key: str):
    if category_key in MODEL_CACHE:
        return MODEL_CACHE[category_key]
    hf_id = get_best_hf_id(category_key)
    if not hf_id:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if "causal" in hf_id.lower() or "llama" in hf_id.lower() or "falcon" in hf_id.lower():
            model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto", torch_dtype=torch.float16)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, device_map="auto", torch_dtype=torch.float16)
        MODEL_CACHE[category_key] = (tokenizer, model)
        return tokenizer, model
    except Exception:
        return None

def load_pipeline_task(task_name: str, hf_id: str):
    if task_name in MODEL_CACHE:
        return MODEL_CACHE[task_name]
    try:
        p = pipeline(task_name, model=hf_id, tokenizer=hf_id, device=0 if torch.cuda.is_available() else -1)
        MODEL_CACHE[task_name] = p
        return p
    except Exception:
        return None

def load_video_pipeline(model_id: str):
    if VIDEO_CACHE_KEY in MODEL_CACHE:
        return MODEL_CACHE[VIDEO_CACHE_KEY]
    try:
        if "ltx2" in model_id.lower():
            from ltx2 import LTX2Pipeline  # adjust to actual class
            pipe = LTX2Pipeline.from_pretrained(model_id, device="cuda" if torch and torch.cuda.is_available() else "cpu")
        elif StableVideoDiffusionPipeline:
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32)
            if torch and torch.cuda.is_available():
                pipe = pipe.to("cuda")
        MODEL_CACHE[VIDEO_CACHE_KEY] = pipe
        return pipe
    except Exception as e:
        print("Video pipeline load failed:", e)
        return None

# ===============================
# Text / NLP Endpoints
# ===============================
@app.post("/text/generate")
async def text_generate(prompt: str = Form(...), category: str = Form("text:chat"), max_tokens: int = Form(512)):
    is_safe, reason = moderate_text(prompt)
    if not is_safe:
        return {"error": reason}
    model_data = load_text_model(category)
    if not model_data:
        return {"error": f"Model {category} not available"}
    tokenizer, model = model_data
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}

@app.post("/text/summarize")
async def text_summarize(text: str = Form(...)):
    hf_id = get_best_hf_id("text:summarize")
    pipe = load_pipeline_task("summarization", hf_id)
    if not pipe:
        return {"error": "Summarization model unavailable"}
    summary = pipe(text)
    return {"summary": summary[0]['summary_text']}

@app.post("/text/qa")
async def text_qa(question: str = Form(...), context: str = Form(...)):
    hf_id = get_best_hf_id("text:qa")
    pipe = load_pipeline_task("question-answering", hf_id)
    if not pipe:
        return {"error": "QA model unavailable"}
    answer = pipe(question=question, context=context)
    return answer

@app.post("/text/translate")
async def text_translate(text: str = Form(...), src_lang: str = Form("en"), tgt_lang: str = Form("fr")):
    hf_id = get_best_hf_id("text:translate")
    pipe = load_pipeline_task("translation", hf_id)
    if not pipe:
        return {"error": "Translation model unavailable"}
    trans = pipe(text)
    return {"translation": trans[0]['translation_text']}

# ===============================
# Code Endpoints
# ===============================
@app.post("/code/generate")
async def code_generate(prompt: str = Form(...), category: str = Form("code:gen"), max_tokens: int = Form(256)):
    model_data = load_text_model(category)
    if not model_data:
        return {"error": f"Code model {category} unavailable"}
    tokenizer, model = model_data
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"code": code}

@app.post("/code/summarize")
async def code_summarize(code: str = Form(...)):
    hf_id = get_best_hf_id("code:summarize")
    pipe = load_pipeline_task("summarization", hf_id)
    if not pipe:
        return {"error": "Code summarization unavailable"}
    summary = pipe(code)
    return {"summary": summary[0]['summary_text']}

# ===============================
# HF Inference helpers (sync + async)
# ===============================
async def hf_inference_async(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set for inference call")
    if not httpx:
        raise RuntimeError("httpx not installed")
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
        raise

def hf_inference_request(model_id: str, inputs, params: dict = None, timeout: int = 120):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set for inference call")
    if not httpx:
        raise RuntimeError("httpx not installed")
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
        raise

# ===============================
# Vision / Image Endpoints (advanced)
# ===============================
IMAGES_DIR = os.getenv("IMAGES_DIR", "/tmp/generated_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

@app.post("/vision/pose")
async def vision_pose(file: UploadFile = File(...)):
    """
    Pose estimation using Detectron2 if installed,
    else returns bounding-box keypoints via HF if available.
    """
    ok, reason = moderate_text(file.filename or "")
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked")

    content = await file.read()

    # Try local Detectron2
    try:
        import numpy as np
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.utils.visualizer import Visualizer

        if "vision:pose" not in MODEL_CACHE:
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
            )
            MODEL_CACHE["vision:pose"] = DefaultPredictor(cfg)

        predictor = MODEL_CACHE["vision:pose"]
        img = Image.open(io.BytesIO(content)).convert("RGB")
        arr = np.array(img)
        outputs = predictor(arr)
        v = Visualizer(arr[:, :, ::-1])
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        buffer = io.BytesIO()
        Image.fromarray(v.get_image()[:, :, ::-1]).save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    except Exception:
        # Fallback: Hugging Face inference if HF_TOKEN and model_id exist
        model_id = get_best_hf_id("vision:pose")
        if HF_TOKEN and model_id and httpx:
            try:
                # async HF inference helper
                async def hf_inference_async(model_id, image_bytes):
                    url = f"https://api-inference.huggingface.co/models/{model_id}"
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, content=image_bytes, headers=headers)
                        resp.raise_for_status()
                        return resp.json()
                resp = await hf_inference_async(model_id, content)
                return {"source": "hf", "result": resp}
            except Exception as e:
                return {"error": "Pose model not available locally or via HF", "detail": str(e)}

        return {"error": "Detectron2 not installed and HF fallback unavailable"}

@app.post("/vision/ocr")
async def vision_ocr(file: UploadFile = File(...)):
    content = await file.read()
    hf_id = get_best_hf_id("vision:ocr")
    # Try pipeline
    try:
        pipe = load_pipeline_task("ocr", hf_id) or load_pipeline_task("image-to-text", hf_id)
        if pipe:
            img = Image.open(io.BytesIO(content))
            res = pipe(img)
            return {"text": res}
    except Exception:
        pass
    # HF inference fallback
    if HF_TOKEN and hf_id and httpx:
        try:
            resp = await hf_inference_async(hf_id, content)
            return {"source": "hf", "result": resp}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="OCR unavailable")

@app.post("/image/inpaint")
async def image_inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form("")):
    img_bytes = await image.read()
    mask_bytes = await mask.read()
    hf_id = get_best_hf_id("image:inpaint")
    # local diffusers inpainting
    if StableDiffusionPipeline and hf_id:
        try:
            key = f"inpaint:{hf_id}"
            if key not in MODEL_CACHE:
                MODEL_CACHE[key] = StableDiffusionPipeline.from_pretrained(hf_id, torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32)
                if torch and torch.cuda.is_available():
                    MODEL_CACHE[key] = MODEL_CACHE[key].to("cuda")
            pipe = MODEL_CACHE[key]
            init_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
            result = pipe(prompt=prompt, image=init_img, mask_image=mask_img)
            out_img = result.images[0]
            out_path = os.path.join(IMAGES_DIR, f"{uuid.uuid4().hex}.png")
            out_img.save(out_path)
            return {"path": out_path}
        except Exception as e:
            # fallback to HF
            pass
    if HF_TOKEN and hf_id:
        try:
            resp = await hf_inference_async(hf_id, {"image": img_bytes, "mask": mask_bytes, "prompt": prompt})
            return {"source": "hf", "result": resp}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=503, detail="Inpainting unavailable")

@app.post("/image/upscale")
async def image_upscale(file: UploadFile = File(...), scale: int = Form(2)):
    content = await file.read()
    hf_id = get_best_hf_id("image:upscale")
    # try HF
    if HF_TOKEN and hf_id:
        try:
            resp = await hf_inference_async(hf_id, content, params={"scale": scale})
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    # local Real-ESRGAN could be added here
    raise HTTPException(status_code=503, detail="Upscaler unavailable")

@app.post("/image/bg_remove")
async def image_bg_remove(file: UploadFile = File(...)):
    content = await file.read()
    hf_id = get_best_hf_id("image:bg_remove")
    if HF_TOKEN and hf_id:
        try:
            resp = await hf_inference_async(hf_id, content)
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=503, detail="Background removal unavailable")

@app.post("/vision/caption")
async def vision_caption(file: UploadFile = File(...)):
    content = await file.read()
    hf_id = get_best_hf_id("vision:caption")
    try:
        pipe = load_pipeline_task("image-captioning", hf_id)
        if pipe:
            img = Image.open(io.BytesIO(content))
            out = pipe(img)
            # pipeline returns list of captions
            if isinstance(out, list) and out:
                return {"caption": out[0].get("caption") or out[0].get("generated_text") or out}
            return {"caption": out}
    except Exception:
        pass
    if HF_TOKEN and hf_id:
        try:
            resp = await hf_inference_async(hf_id, content)
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=503, detail="Caption unavailable")

@app.post("/image/generate")
async def image_generate(prompt: str = Form(...), samples: int = Form(1)):
    """
    Try OpenAI DALL·E 3 (gpt-image-1) if OPENAI_API_KEY present; otherwise use HF image model (SDXL) if available.
    """
    ok, reason = moderate_text(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked")

    # OpenAI DALL·E 3
    if OPENAI_API_KEY:
        try:
            import base64
            payload = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": samples,
                "size": "1024x1024",
                "response_format": "b64_json"
            }
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            r = httpx.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=90.0)
            r.raise_for_status()
            jr = r.json()
            data = jr.get("data", [])
            urls = []
            for d in data:
                b64 = d.get("b64_json")
                if b64:
                    img_bytes = base64.b64decode(b64)
                    out_path = os.path.join(IMAGES_DIR, f"{uuid.uuid4().hex}.png")
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    urls.append(out_path)
            if urls:
                return {"provider": "openai", "images": urls}
        except Exception as e:
            # fallback to HF
            pass

    # HF SDXL
    hf_id = get_best_hf_id("image:sdxl") or get_best_hf_id("image:txt2img")
    if StableDiffusionPipeline and hf_id:
        try:
            key = f"sdxl:{hf_id}"
            if key not in MODEL_CACHE:
                MODEL_CACHE[key] = StableDiffusionPipeline.from_pretrained(hf_id, torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32)
                if torch and torch.cuda.is_available():
                    MODEL_CACHE[key] = MODEL_CACHE[key].to("cuda")
            pipe = MODEL_CACHE[key]
            outs = []
            for _ in range(max(1, samples)):
                res = pipe(prompt)
                img = res.images[0]
                out_path = os.path.join(IMAGES_DIR, f"{uuid.uuid4().hex}.png")
                img.save(out_path)
                outs.append(out_path)
            return {"provider": "sdxl", "images": outs}
        except Exception as e:
            return {"error": "Image generation failed", "detail": str(e)}

    # HF inference fallback
    if HF_TOKEN and hf_id and httpx:
        try:
            resp = await hf_inference_async(hf_id, {"prompt": prompt, "num_images": samples})
            return {"source": "hf", "result": resp}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=503, detail="No image generator available")

# ===============================
# Audio / Speech Endpoints
# ===============================
@app.post("/speech/tts")
async def speech_tts(text: str = Form(...), voice: Optional[str] = Form(None), fmt: str = Form("mp3")):
    ok, reason = moderate_text(text)
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked")
    # ElevenLabs preferred if key present
    if ELEVEN_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice or 'alloy'}"
            headers = {"Accept": "audio/mpeg", "xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
            payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    out_path = f"/tmp/tts_{uuid.uuid4().hex}.{fmt}"
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    return FileResponse(out_path, media_type="audio/mpeg")
        except Exception:
            pass
    # Coqui TTS local fallback
    hf_id = get_best_hf_id("speech:tts")
    if CoquiTTS and hf_id:
        try:
            if hf_id not in MODEL_CACHE:
                MODEL_CACHE[hf_id] = CoquiTTS(model_name=hf_id)
            tts = MODEL_CACHE[hf_id]
            out_path = f"/tmp/tts_{uuid.uuid4().hex}.{fmt}"
            tts.tts_to_file(text=text, speaker=voice or None, file_path=out_path)
            return FileResponse(out_path, media_type="audio/mpeg")
        except Exception as e:
            return {"error": str(e)}
    # HF fallback
    if HF_TOKEN and httpx:
        model_id = get_best_hf_id("speech:tts")
        try:
            resp = await hf_inference_async(model_id, text)
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=503, detail="TTS unavailable")

@app.post("/speech/stt")
async def speech_stt(file: UploadFile = File(...)):
    tmp = f"/tmp/stt_{uuid.uuid4().hex}_{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    # faster-whisper
    if WhisperModel:
        try:
            key = "whisper:local"
            if key not in MODEL_CACHE:
                MODEL_CACHE[key] = WhisperModel(get_best_hf_id("speech:whisper") or "large")
            whisper = MODEL_CACHE[key]
            segments, info = whisper.transcribe(tmp)
            text = " ".join([s.text for s in segments])
            return {"source": "local", "text": text}
        except Exception:
            pass
    # HF fallback
    hf_id = get_best_hf_id("speech:whisper")
    if HF_TOKEN and hf_id and httpx:
        try:
            with open(tmp, "rb") as fh:
                content = fh.read()
            resp = await hf_inference_async(hf_id, content)
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    raise HTTPException(status_code=503, detail="STT unavailable")

@app.post("/speech/voice_clone")
async def speech_voice_clone(seed_audio: UploadFile = File(...), text: str = Form(...)):
    # Placeholder for ElevenLabs voice cloning or YourTTS if available
    if ELEVEN_API_KEY:
        return {"note": "Implement ElevenLabs voice cloning using ELEVEN_API_KEY"}
    return {"error": "Voice cloning requires an external API or local voice model"}

@app.post("/audio/music")
async def audio_music(prompt: str = Form(...), duration: int = Form(20)):
    model_id = get_best_hf_id("audio:musicgen")
    if HF_TOKEN and model_id and httpx:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt, "duration": duration})
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Music generation unavailable"}

# ===============================
# Video Endpoints
# ===============================
VIDEO_CACHE_KEY = "video_pipeline"

def load_video_pipeline(model_id: str):
    if VIDEO_CACHE_KEY in MODEL_CACHE:
        return MODEL_CACHE[VIDEO_CACHE_KEY]
    if not StableVideoDiffusionPipeline:
        return None
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32)
        if torch and torch.cuda.is_available():
            pipe = pipe.to("cuda")
        MODEL_CACHE[VIDEO_CACHE_KEY] = pipe
        return pipe
    except Exception:
        return None

@app.post("/video/generate")
async def video_generate(
    prompt: str = Form(...),
    seconds: int = Form(4),
    fps: int = Form(8),
    num_inference_steps: int = Form(30),
):
    """
    Generate a short video from text using multiple backends:
    1. StableVideoDiffusionPipeline (local)
    2. Hugging Face Inference API
    3. DALL·E 3 (OpenAI)
    """
    # ---- moderation ----
    ok, reason = moderate_text(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked")

    # ---- local Stable Video Diffusion ----
    model_id = get_best_hf_id("video:img2vid") or get_best_hf_id("video:txt2vid")
    if StableVideoDiffusionPipeline and model_id:
        pipe = load_video_pipeline(model_id)
        if pipe:
            try:
                frames_count = max(8, int(seconds * fps))
                result = pipe(prompt, num_inference_steps=num_inference_steps, num_frames=frames_count)
                video_frames = getattr(result, "frames", None) or result[0]

                import imageio
                import numpy as np
                video_frames = [np.array(f).astype(np.uint8) for f in video_frames]

                out_vid = f"/tmp/video_{uuid.uuid4().hex}.mp4"
                imageio.mimwrite(out_vid, video_frames, fps=fps)
                return FileResponse(out_vid, media_type="video/mp4")
            except Exception as e:
                print("Local video generation failed:", e)

    # ---- HF Inference API fallback ----
    if HF_TOKEN and httpx and model_id:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt, "seconds": seconds, "fps": fps})
            return {"source": "hf_inference", "result": resp}
        except Exception as e:
            print("HF Inference video failed:", e)

    # ---- DALL·E 3 fallback via OpenAI ----
    if OPENAI_MOD:
        try:
            # OpenAI API currently supports image generation; we simulate video by generating frames
            frames_count = max(4, int(seconds * fps))
            frame_urls = []
            for i in range(frames_count):
                response = OPENAI_MOD.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024"
                )
                url = response.data[0].url
                frame_urls.append(url)
            return {"source": "dall-e-3", "frames": frame_urls, "fps": fps}
        except Exception as e:
            print("DALL·E 3 fallback failed:", e)

    # ---- nothing available ----
    return {"error": "Video generation unavailable — no backend succeeded"}
    
@app.post("/video/img2vid")
async def video_img2vid(seed_image: UploadFile = File(...), prompt: str = Form(...), frames: int = Form(16)):
    content = await seed_image.read()
    model_id = get_best_hf_id("video:img2vid")
    if StableVideoDiffusionPipeline and model_id:
        pipe = load_video_pipeline(model_id)
        if not pipe:
            return {"error": "Video pipeline not available"}
        try:
            init_img = Image.open(io.BytesIO(content)).convert("RGB")
            res = pipe(init_img, num_inference_steps=30, num_frames=frames)
            video_frames = getattr(res, "frames", None) or res[0]
            out_vid = f"/tmp/video_{uuid.uuid4().hex}.mp4"
            try:
                import imageio
                imageio.mimwrite(out_vid, video_frames, fps=8)
                return FileResponse(out_vid, media_type="video/mp4")
            except Exception:
                return {"frames_count": len(video_frames)}
        except Exception as e:
            return {"error": str(e)}
    # HF fallback
    if HF_TOKEN and model_id and httpx:
        try:
            resp = await hf_inference_async(model_id, {"prompt": prompt})
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "img2vid unavailable"}

# ===============================
# 3D / Geometry Endpoints
# ===============================
@app.post("/3d/generate")
async def generate_3d(prompt: str = Form(...)):
    # Moderate input
    ok, reason = moderate_text(prompt)
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked by moderation")

    hf_id = get_best_hf_id("3d:object")
    if not hf_id:
        return {"error": "No 3D model configured"}

    # Attempt local Point-E generation
    try:
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, model_from_config
        from point_e.diffusion.sampling import sample_model
        from point_e.util.point_cloud import save_point_cloud

        cache_key = f"3d:{hf_id}"
        if cache_key not in MODEL_CACHE:
            device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
            cfg = DIFFUSION_CONFIGS['base']
            model = model_from_config(cfg, device=device)
            MODEL_CACHE[cache_key] = model
        model = MODEL_CACHE[cache_key]

        # Sample point cloud
        pc = sample_model(model, prompt)
        out_path = f"/tmp/pointcloud_{uuid.uuid4().hex}.ply"
        save_point_cloud(pc, out_path)
        return FileResponse(out_path)
    
    except ImportError:
        # Point-E not installed
        pass
    except Exception as e:
        return {"error": f"Local Point-E failed: {e}"}

    # Hugging Face Inference API fallback
    if HF_TOKEN and hf_id and httpx:
        try:
            async def hf_inference_async(model_id: str, prompt: str):
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(
                        f"https://api-inference.huggingface.co/models/{model_id}",
                        headers=headers,
                        json={"inputs": prompt}
                    )
                    r.raise_for_status()
                    return r.json()
            result = await hf_inference_async(hf_id, prompt)
            return {"source": "hf_inference", "result": result}

        except Exception as e:
            return {"error": f"HF Inference failed: {e}"}

    return {"error": "3D generation unavailable"}

# ===============================
# Reinforcement Learning / Decision endpoints
# ===============================
import json
import numpy as np

@app.post("/rl/predict")
async def rl_predict(obs: str = Form(...), model: str = Form("rl:policy")):
    """
    Parses a JSON or CSV observation string and returns a dummy action.
    Can be replaced with stable-baselines3 / RL model inference.
    """
    hf_id = get_best_hf_id(model)
    
    # Attempt to parse JSON first
    try:
        obs_data = json.loads(obs)
        if isinstance(obs_data, dict):
            obs_array = np.array(list(obs_data.values()), dtype=np.float32)
        elif isinstance(obs_data, list):
            obs_array = np.array(obs_data, dtype=np.float32)
        else:
            obs_array = np.array([float(obs_data)], dtype=np.float32)
    except Exception:
        # Fallback: try CSV-style parsing
        try:
            obs_array = np.array([float(x.strip()) for x in obs.split(",")], dtype=np.float32)
        except Exception:
            obs_array = np.array([0.0], dtype=np.float32)

    # Dummy action: just a random float array with same shape
    action = np.random.randn(*obs_array.shape).tolist()

    return {
        "action": action,
        "model": hf_id,
        "obs_snippet": str(obs_array.tolist())[:200]
    }
# ===============================
# ML Utilities
# ===============================
@app.post("/ml/anomaly")
async def ml_anomaly(file: UploadFile = File(...)):
    content = await file.read()
    hf_id = get_best_hf_id("ml:anomaly")

    # Try local CLIP
    try:
        from transformers import CLIPProcessor, CLIPModel

        key = f"clip:{hf_id}"
        if key not in MODEL_CACHE:
            processor = CLIPProcessor.from_pretrained(hf_id)
            model = CLIPModel.from_pretrained(hf_id)
            MODEL_CACHE[key] = (processor, model)
        processor, model = MODEL_CACHE[key]

        img = Image.open(io.BytesIO(content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        if torch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            score = float(torch.norm(emb).cpu().item())
        else:
            score = None

        return {"source": "local", "anomaly_score": score}

    except Exception as e_local:
        # HF fallback
        if HF_TOKEN and hf_id and httpx:
            try:
                resp = await hf_inference_async(hf_id, content)
                return {"source": "hf", "result": resp}
            except Exception as e_hf:
                return {"error": f"HF fallback failed: {str(e_hf)}"}

    return {"error": "Anomaly detection unavailable"}
    
@app.post("/ml/recommend")
async def ml_recommend(user_id: str = Form(...), context: Optional[str] = Form(None)):
    # placeholder: return empty recommendations
    return {"user_id": user_id, "recommendations": [], "note": "Connect real recommender"}

# ===============================
# Medical / Scientific Endpoints
# ===============================
@app.post("/medical/scan")
async def medical_scan(file: UploadFile = File(...), model: str = Form("medical:imaging")):
    content = await file.read()
    hf_id = get_best_hf_id(model)
    # Very explicit: DO NOT USE FOR REAL DIAGNOSIS
    # Try HF inference if available
    if HF_TOKEN and hf_id and httpx:
        try:
            resp = await hf_inference_async(hf_id, content)
            return {"source": "hf", "result": resp}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Medical imaging model unavailable"}

@app.post("/medical/protein")
async def medical_protein(
    sequence: str = Form(...),
    model: str = Form("medical:protein")
):
    hf_id = get_best_hf_id(model)
    if not hf_id:
        return {"error": "Protein model not found"}

    # Try ESM2 via transformers
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        key = f"esm:{hf_id}"
        if key not in MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model_obj = AutoModel.from_pretrained(hf_id)
            model_obj.eval()  # ensure evaluation mode
            MODEL_CACHE[key] = (tokenizer, model_obj)

        tokenizer, model_obj = MODEL_CACHE[key]
        inputs = tokenizer(sequence, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model_obj.to(device)

        with torch.no_grad():
            outputs = model_obj(**inputs)
            # mean pooling over sequence length
            emb = outputs.last_hidden_state.mean(dim=1)
        return {
            "source": "local",
            "embedding_norm": float(torch.norm(emb).item())
        }

    except Exception as e_local:
        # fallback: HF Inference API if token available
        if HF_TOKEN and hf_id and httpx:
            try:
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"https://api-inference.huggingface.co/models/{hf_id}",
                        headers=headers,
                        json={"inputs": sequence}
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return {"source": "hf_inference", "result": data}
            except Exception as e_hf:
                return {"error": f"HF fallback failed: {str(e_hf)}"}

        return {"error": f"Protein model unavailable locally: {str(e_local)}"}
        
# ===============================
# Ask endpoint (natural language routing)
# ===============================
@app.post("/ask")
async def ask(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form("guest")
):
    # ---- 1. Moderate input ----
    content_to_check = text or (file.filename if file else "")
    ok, reason = moderate_text(content_to_check)
    if not ok:
        raise HTTPException(status_code=400, detail=reason or "Blocked")

    # ---- 2. Detect type ----
    category = None
    if file:
        ext = file.filename.split(".")[-1].lower()
        if ext in ["png", "jpg", "jpeg", "bmp"]:
            category = "image"
        elif ext in ["mp3", "wav", "ogg"]:
            category = "audio"
        elif ext in ["mp4", "mov", "avi"]:
            category = "video"
        else:
            category = "file"
    elif text:
        # Keyword-based fallback; replace with embedding classifier if desired
        lower = text.lower()
        if any(k in lower for k in ["code", "function", "script", "program"]):
            category = "code"
        elif any(k in lower for k in ["image", "draw", "generate", "painting", "picture"]):
            category = "image"
        elif any(k in lower for k in ["video", "movie", "clip"]):
            category = "video"
        elif any(k in lower for k in ["music", "song", "audio", "tts"]):
            category = "audio"
        elif any(k in lower for k in ["3d", "mesh", "point cloud", "nerf"]):
            category = "3d"
        else:
            category = "text"
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    # ---- 3. Call the appropriate handler ----
    try:
        if category == "text":
            res = await text_generate(prompt=text, category="text:chat", max_tokens=256)
        elif category == "code":
            res = await code_generate(prompt=text, category="code:gen", max_tokens=256)
        elif category == "image":
            if file:
                res = await vision_caption(file=file)
            else:
                res = await image_generate(prompt=text, samples=1)
        elif category == "video":
            res = await video_generate(prompt=text, seconds=4)
        elif category == "audio":
            if file:
                res = await speech_stt(file=file)
            else:
                res = await speech_tts(text=text)
        elif category == "3d":
            res = await generate_3d(prompt=text)
        else:
            res = {"error": "Cannot classify request"}
    except Exception as e:
        res = {"error": str(e)}

    return {"type": category, "result": res}
    
# ===============================
# WebSocket streaming chat (improved with chunked streaming)
# ===============================
import asyncio  # ensure imported

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            prompt = data.get("prompt", "")
            category = data.get("category", "text:chat")
            ok, reason = moderate_text(prompt)
            if not ok:
                await ws.send_json({"error": reason or "Blocked"})
                continue
            model_data = load_text_model(category)
            if not model_data:
                if HF_TOKEN and httpx:
                    model_id = get_best_hf_id(category)
                    try:
                        resp = await hf_inference_async(model_id, prompt)
                        out_text = resp.get("generated_text") or str(resp)
                        await ws.send_json({"text": out_text})
                        continue
                    except Exception as e:
                        await ws.send_json({"error": str(e)})
                        continue
                await ws.send_json({"error": f"Model {category} unavailable"})
                continue
            tokenizer, model = model_data
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=256)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # stream in 120-char chunks
                for i in range(0, len(text), 120):
                    await ws.send_json({"delta": text[i:i+120]})
                    await asyncio.sleep(0.03)
                await ws.send_json({"done": True, "final": text})
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
# ===============================
# Admin / Utilities / Search / Weather / Wolfram
# ===============================
@app.get("/admin/models")
def admin_models():
    return {"registry_keys": list(MODEL_REGISTRY.keys()), "loaded_models": list(MODEL_CACHE.keys())}

@app.post("/admin/clear_cache")
def admin_clear_cache():
    MODEL_CACHE.clear()
    return {"cleared": True}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase": bool(supabase),
        "redis": bool(redis_client),
        "torch": bool(torch),
        "hf_token": bool(HF_TOKEN),
        "openai_moderation": bool(OPENAI_MOD),
        "loaded_models": list(MODEL_CACHE.keys())
    }

@app.get("/search")
async def google_search(q: str = Query(...)):
    if not SERPAPI_KEY or not httpx:
        raise HTTPException(status_code=503, detail="SerpAPI key or httpx not configured")
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": q, "api_key": SERPAPI_KEY}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        results = []
        for r in data.get("organic_results", [])[:5]:
            results.append({"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")})
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather")
async def get_weather(city: str = Query(...)):
    if not OPENWEATHER_KEY or not httpx:
        raise HTTPException(status_code=503, detail="OpenWeather API key or httpx not set")
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return {"city": city, "temperature_c": data["main"]["temp"], "humidity": data["main"]["humidity"], "description": data["weather"][0]["description"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wolfram")
async def wolfram_query(query: str = Query(...)):
    if not WOLFRAM_KEY or not httpx:
        raise HTTPException(status_code=503, detail="Wolfram API key or httpx not set")
    url = "http://api.wolframalpha.com/v2/query"
    params = {"input": query, "appid": WOLFRAM_KEY, "format": "plaintext"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        pods = root.findall(".//pod")
        results = []
        for pod in pods:
            title = pod.attrib.get("title", "")
            plaintexts = [pt.text for pt in pod.findall(".//plaintext") if pt.text]
            if plaintexts:
                results.append({"title": title, "content": plaintexts})
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# Upload / Library
# ===============================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".bin"
    out = f"{IMAGES_DIR}/{uuid.uuid4().hex}{ext}"
    with open(out, "wb") as f:
        f.write(await file.read())
    public_url = None
    if supabase:
        try:
            bucket = "generated_media"
            fname = os.path.basename(out)
            with open(out, "rb") as fh:
                supabase.storage.from_(bucket).upload(fname, fh, {"upsert": True})
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{fname}"
        except Exception:
            public_url = None
    return {"path": out, "public_url": public_url}

@app.get("/library")
def library(page: int = 0, page_size: int = 24):
    items = []
    try:
        files = sorted(os.listdir(IMAGES_DIR), reverse=True)
        start = page * page_size
        for f in files[start:start + page_size]:
            items.append({"file": f, "path": os.path.join(IMAGES_DIR, f)})
    except Exception:
        pass
    return {"items": items}

# ===============================
# Startup
# ===============================
@app.on_event("startup")
def on_startup():
    print(f"🚀 {APP_NAME} starting up — created by {CREATOR}")
    print("Configured components:")
    print(" - Supabase:", bool(supabase))
    print(" - Redis:", bool(redis_client))
    print(" - HF_TOKEN:", bool(HF_TOKEN))
    print(" - Torch:", bool(torch))
    print(" - Diffusers:", bool(StableDiffusionPipeline))
    print(" - Whisper:", bool(WhisperModel))
    try:
        if AutoTokenizer:
            _ = AutoTokenizer.from_pretrained("google/flan-t5-small")
            print("✅ Warmed small tokenizer")
    except Exception:
        pass
    os.makedirs(IMAGES_DIR, exist_ok=True)

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
