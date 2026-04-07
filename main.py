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
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() if os.getenv("GROQ_API_KEY") else ""
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL")
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("heloxai-server")

app = FastAPI(title="HeloXAI Multimodal Server", redirect_slashes=False)

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
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-70b-versatile")
CODE_MODEL = os.getenv("CODE_MODEL", "llama-3.1-70b-versatile")
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

# ---------- Request/Response Models ----------
class UniversalRequest(BaseModel):
    prompt: str = Field(..., description="The prompt or query")
    context: Optional[str] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    model: Optional[str] = Field(None, description="Model to use")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(4096, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    tools: Optional[List[str]] = Field(None, description="Tools to use")
    files: Optional[List[str]] = Field(None, description="Files to process")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class UniversalResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    processing_time: float
    tools_used: List[str]
    metadata: Optional[Dict[str, Any]] = None

# ---------- Helper Functions ----------
def get_groq_headers() -> dict:
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def count_tokens(text: str) -> int:
    return len(text) // 4

def detect_language(prompt: str) -> str:
    language_map = {
        "python": ["python", "py", "django", "flask"],
        "javascript": ["javascript", "js", "node", "react", "typescript", "ts"],
        "java": ["java", "spring"],
        "cpp": ["c++", "cpp"],
        "go": ["golang", "go lang"],
        "rust": ["rust", "rustlang"],
        "sql": ["sql", "mysql", "postgres"],
    }
    prompt_lower = prompt.lower()
    for lang, keywords in language_map.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                return lang
    return "python"

def detect_intent(prompt: str) -> tuple:
    if not prompt:
        return ("chat", 0.0)
    
    prompt_lower = prompt.lower().strip()
    intent_patterns = {
        "image_generation": [r"(generate|create|make|draw)\s+(a\s+)?(image|picture|photo|art)", r"dall[eé]", r"stable\s+diffusion"],
        "video_generation": [r"(generate|create|make)\s+(a\s+)?(video|clip|animation)"],
        "vision_analysis": [r"(analyze|describe|what(?:'s| is)\s+in)\s+(this\s+)?(image|picture)"],
        "math_calculation": [r"(calculate|compute|solve|evaluate)\s+", r"\d+\s*[\+\-\*\/]\s*\d+"],
        "joke": [r"tell\s+(me\s+)?(a\s+)?joke", r"make\s+me\s+laugh"],
        "code_generation": [r"(write|create|generate|code|implement)\s+(a\s+)?(function|class|program|script)", r"(python|javascript|java)\s+code"],
        "web_search": [r"(search|look\s+up|find|google)\s+", r"(what\s+is|who\s+is)\s+"],
        "text_to_speech": [r"(speak|say|read|narrate)\s+", r"text\s+to\s+speech", r"tts"],
        "audio_transcription": [r"transcribe\s+(this\s+)?(audio|voice)", r"speech\s+to\s+text"],
        "translation": [r"translate\s+", r"in\s+(spanish|french|german|japanese|chinese)"],
        "summarization": [r"summarize\s+", r"(summary|tldr|tl;dr)"],
        "reasoning": [r"(reason|think)\s+(through|step\s+by\s+step)", r"explain\s+(why|how)"],
        "creative_writing": [r"(write|create)\s+(a\s+)?(story|poem|song|lyrics)"],
    }
    
    best_intent = "chat"
    best_score = 0
    
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                position_weight = 1.0 - (match.start() / max(len(prompt), 1)) * 0.5
                score = 0.85 * position_weight
                if score > best_score:
                    best_score = score
                    best_intent = intent
    
    return (best_intent, min(best_score, 1.0))

def get_elevenlabs_voice_id(voice_name: str) -> str:
    voice_map = {
        "alloy": "21m00Tcm4TlvDq8ikWAM", "rachel": "21m00Tcm4TlvDq8ikWAM",
        "adam": "pNInz6obpgDQGcFmaJgB", "antoni": "ErXwobaYiN019PkySvjV",
        "bella": "EXAVITQu4vr4xnSDxMaL", "callum": "yoZ06aMxZJJ28mfd3POQ",
        "charlotte": "XB0fDUnXU5powFXDhCwa", "daniel": "onwK4e9ZLuTAKqWW03F9",
        "emily": "LcfcDJNUP1GQjkzn1rUU", "finn": "TxGEqnHWrfWFTfGW9XjX",
        "grace": "oWAxZDx7w5VEj9dCyTzz", "jenny": "SAz9YHcvj6GT2YYXdXww",
    }
    return voice_map.get(voice_name.lower(), "21m00Tcm4TlvDq8ikWAM")

async def extract_document_text(doc: str) -> str:
    try:
        if doc.startswith("http"):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(doc)
                resp.raise_for_status()
                data = resp.content
        else:
            if "," in doc:
                doc = doc.split(",", 1)[1]
            data = base64.b64decode(doc)
        
        try:
            doc_pdf = fitz.open(stream=data, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc_pdf])
            if text.strip():
                return text
        except:
            pass
        
        try:
            doc_docx = docx.Document(io.BytesIO(data))
            text = "\n".join([p.text for p in doc_docx.paragraphs])
            if text.strip():
                return text
        except:
            pass
        
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error: {str(e)}]"

async def duckduckgo_search(query: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            data = r.json()
            results = []
            if data.get("Abstract"):
                results.append({"title": data["Heading"], "snippet": data["Abstract"], "url": data["AbstractURL"]})
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({"title": topic.get("FirstURL", ""), "snippet": topic["Text"], "url": topic.get("FirstURL", "")})
            return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def wolfram_alpha_query(query: str) -> str:
    if not WOLFRAM_ALPHA_API_KEY:
        return "Wolfram Alpha not configured"
    try:
        import wolframalpha
        client = wolframalpha.Client(WOLFRAM_ALPHA_API_KEY)
        res = client.query(query)
        return next(res.results).text
    except Exception as e:
        return f"Error: {str(e)}"

async def tell_joke(category: str) -> dict:
    joke_prompt = f"Tell a funny {category} joke. Return only the joke, no introduction."
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
            "model": CHAT_MODEL, "messages": [{"role": "user", "content": joke_prompt}],
            "max_tokens": 200, "temperature": 0.9
        })
        return {"joke": r.json()["choices"][0]["message"]["content"]}

async def solve_math(prompt: str) -> dict:
    if WOLFRAM_ALPHA_API_KEY:
        result = wolfram_alpha_query(prompt)
        return {"answer": result, "method": "wolfram"}
    
    math_prompt = f"Solve this math problem step by step:\n{prompt}\n\nProvide the final answer clearly."
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
            "model": CHAT_MODEL, "messages": [{"role": "user", "content": math_prompt}],
            "max_tokens": 1000, "temperature": 0.2
        })
        return {"answer": r.json()["choices"][0]["message"]["content"], "method": "llm"}

async def chat_with_tools(user_id: str, messages: list) -> str:
    """Main chat function - can be extended with tool calling"""
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
            "model": CHAT_MODEL, "messages": messages,
            "max_tokens": 4096, "temperature": 0.7
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ---------- Memory System ----------
async def fetch_user_memory(user_id: str) -> str:
    if not user_id or user_id == "anonymous":
        return ""
    try:
        res = await asyncio.to_thread(
            lambda: supabase.table("user_memories")
            .select("category, content")
            .eq("user_id", user_id)
            .order("last_referenced", desc=True)
            .limit(10)
            .execute()
        )
        if not res.data:
            return ""
        memory_str = "User Profile & Memories:\n"
        for mem in res.data:
            memory_str += f"- [{mem.get('category', 'info')}] {mem['content']}\n"
        return memory_str
    except Exception as e:
        logger.error(f"Failed to fetch user memory: {e}")
        return ""

async def extract_and_save_memory(user_id: str, prompt: str, response: str):
    if not user_id or user_id == "anonymous":
        return
    try:
        extraction_prompt = f"""Extract facts about the USER from this conversation. Return JSON array of {{"category", "content"}} or [].
User: {prompt[:500]}
AI: {response[:500]}"""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": extraction_prompt}],
                "temperature": 0.1, "max_tokens": 200
            })
            raw_text = r.json()["choices"][0]["message"]["content"]
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            facts = json.loads(raw_text)
            for fact in facts:
                if fact.get("content"):
                    await asyncio.to_thread(
                        lambda f=fact: supabase.table("user_memories").upsert({
                            "user_id": user_id, "category": f.get("category", "info"),
                            "content": f["content"], "last_referenced": datetime.utcnow().isoformat()
                        }, on_conflict="user_id,content").execute()
                    )
    except Exception as e:
        logger.warning(f"Memory extraction failed: {e}")

# ---------- Auth (Optional) ----------
async def get_current_user_optional(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Add JWT verification here if needed
        return {"id": token}
    return {}

# ---------- Main Chat Endpoint ----------
@app.post("/ask/universal")
async def ask_universal(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user_optional)
):
    try:
        body = await request.json()
        prompt = (body.get("prompt") or "").strip()
        conversation_id = body.get("conversation_id")
        files = body.get("files", [])
        stream = body.get("stream", True)
        
        output_type = body.get("output_type", "text")
        language = body.get("language")
        target_language = body.get("target_language")
        image_style = body.get("image_style")
        image_size = body.get("image_size", "1024x1024")
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
        user_id = identity.get("id") or request.cookies.get("guest_id") or str(uuid.uuid4())

        if not identity.get("id") and not request.cookies.get("guest_id"):
            response.set_cookie(key="guest_id", value=user_id, httponly=True, secure=True, samesite="lax", max_age=60*60*24*7)

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        await asyncio.to_thread(
            lambda: supabase.table("conversations").upsert({
                "id": conversation_id, "user_id": user_id, "created_at": datetime.utcnow().isoformat()
            }).execute()
        )

        history_res = await asyncio.to_thread(
            lambda: supabase.table("messages").select("role, content")
            .eq("conversation_id", conversation_id).order("created_at").limit(20).execute()
        )
        history_messages = history_res.data or []

        detected_intent, confidence = detect_intent(prompt)
        
        # Override intent based on explicit params
        if output_type == "code":
            detected_intent = "code_generation"
        elif target_language:
            detected_intent = "translation"
        elif enable_cot:
            detected_intent = "reasoning"
        elif execute_code:
            detected_intent = "code_execution"

        # ------------------------- JOKE -------------------------
        if detected_intent == "joke":
            return await tell_joke("general")

        # ------------------------- MATH -------------------------
        elif detected_intent == "math_calculation":
            if enable_cot:
                return await solve_math_with_reasoning(prompt, stream)
            return await solve_math(prompt)

        # ------------------------- CODE GENERATION -------------------------
        elif detected_intent == "code_generation":
            lang = language or detect_language(prompt)
            code_prompt = f"Write a {lang} program for: {prompt}"
            if context:
                code_prompt = f"Context: {context}\n\n{code_prompt}"
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                    "model": CODE_MODEL,
                    "messages": [{"role": "system", "content": "Generate clean, well-documented code. Return ONLY code in a code block."},
                                 {"role": "user", "content": code_prompt}],
                    "max_tokens": max_tokens, "temperature": temperature
                })
                return {"language": lang, "code": r.json()["choices"][0]["message"]["content"]}

        # ------------------------- CODE EXECUTION -------------------------
        elif detected_intent == "code_execution":
            lang = language or detect_language(prompt)
            async def event_generator():
                try:
                    yield sse({"type": "status", "status": "generating"})
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                            "model": CODE_MODEL,
                            "messages": [{"role": "user", "content": f"Write executable {lang} code for: {prompt}"}],
                            "max_tokens": max_tokens
                        })
                        code = r.json()["choices"][0]["message"]["content"]
                    code_match = re.search(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                    yield sse({"type": "code", "language": lang, "code": code})
                    yield sse({"type": "status", "status": "executing"})
                    
                    lang_id = JUDGE0_LANGUAGES.get(lang.lower(), 71)
                    async with httpx.AsyncClient(timeout=30) as client:
                        submit = await client.post(f"{JUDGE0_URL}/submissions", headers={
                            "X-RapidAPI-Key": JUDGE0_KEY, "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
                            "Content-Type": "application/json"
                        }, json={"source_code": code, "language_id": lang_id, "stdin": ""})
                        token = submit.json()["token"]
                        await asyncio.sleep(2)
                        for _ in range(10):
                            result = await client.get(f"{JUDGE0_URL}/submissions/{token}", headers={
                                "X-RapidAPI-Key": JUDGE0_KEY, "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
                            })
                            status_id = result.json().get("status", {}).get("id", 0)
                            if status_id in [1, 2]:
                                await asyncio.sleep(1)
                                continue
                            yield sse({"type": "execution", "exit_code": status_id,
                                       "output": result.json().get("stdout") or result.json().get("stderr", "No output")})
                            break
                    yield sse({"type": "done"})
                except Exception as e:
                    yield sse({"type": "error", "message": str(e)})
            return StreamingResponse(event_generator(), media_type="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

        # ------------------------- SEARCH -------------------------
        elif detected_intent == "web_search":
            query = re.sub(r"^(search for|look up|find|google)\s+", "", prompt.lower(), flags=re.IGNORECASE)
            if stream:
                async def event_generator():
                    try:
                        yield sse({"type": "status", "status": "searching"})
                        results = await duckduckgo_search(query)
                        yield sse({"type": "search_results", "results": results})
                        yield sse({"type": "status", "status": "summarizing"})
                        async with httpx.AsyncClient(timeout=60) as client:
                            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                                "model": CHAT_MODEL,
                                "messages": [{"role": "user", "content": f"Answer: {query}\nResults: {json.dumps(results)}"}],
                                "max_tokens": 2048, "temperature": 0.3
                            })
                            summary = r.json()["choices"][0]["message"]["content"]
                        for char in summary:
                            yield sse({"type": "token", "text": char})
                            await asyncio.sleep(0.005)
                        yield sse({"type": "done"})
                    except Exception as e:
                        yield sse({"type": "error", "message": str(e)})
                return StreamingResponse(event_generator(), media_type="text/event-stream",
                                          headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
            return await duckduckgo_search(query)

        # ------------------------- TTS -------------------------
        elif detected_intent == "text_to_speech":
            text = re.sub(r"[#*`\[\]]", "", prompt).strip()
            if ELEVENLABS_API_KEY:
                try:
                    voice_id = get_elevenlabs_voice_id(voice)
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                            json={"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}})
                        return {"text": text, "audio": base64.b64encode(r.content).decode(), "provider": "elevenlabs"}
                except Exception as e:
                    logger.warning(f"ElevenLabs failed: {e}")
            if OPENAI_API_KEY:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post("https://api.openai.com/v1/audio/speech",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                        json={"model": "tts-1", "voice": voice, "input": text})
                    return {"text": text, "audio": base64.b64encode(r.content).decode(), "provider": "openai"}
            raise HTTPException(500, "No TTS provider available")

        # ------------------------- TRANSLATION -------------------------
        elif detected_intent == "translation":
            text_to_translate = prompt
            for pattern in [r"translate.*?to\s+(\w+):\s*(.+)", r"in\s+(\w+):\s*(.+)", r"to\s+(\w+):\s*(.+)"]:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    target_language = target_language or match.group(1)
                    text_to_translate = match.group(2)
                    break
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": f"Translate to {target_language}:\n{text_to_translate}"}],
                    "max_tokens": max_tokens, "temperature": 0.3
                })
                return {"original": text_to_translate, "translated": r.json()["choices"][0]["message"]["content"], "target_language": target_language}

        # ------------------------- SUMMARIZATION -------------------------
        elif detected_intent == "summarization":
            text_to_summarize = re.sub(r"^summarize[:\s]*", "", prompt, flags=re.IGNORECASE).strip()
            if documents:
                text_to_summarize = "\n\n".join([await extract_document_text(d) for d in documents]) + "\n\n" + text_to_summarize
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": f"Summarize:\n{text_to_summarize}"}],
                    "max_tokens": max_tokens, "temperature": 0.3
                })
                return {"summary": r.json()["choices"][0]["message"]["content"]}

        # ------------------------- REASONING -------------------------
        elif detected_intent == "reasoning":
            reasoning_sys = "You are an expert reasoner. Think through problems systematically with clear step-by-step formatting."
            messages = [{"role": "system", "content": reasoning_sys}] + history_messages[-6:] + [{"role": "user", "content": f"Think step by step:\n\n{prompt}"}]
            if stream:
                async def event_generator():
                    try:
                        yield sse({"type": "status", "status": "thinking"})
                        async with httpx.AsyncClient(timeout=120) as client:
                            async with client.stream("POST", GROQ_URL, headers=get_groq_headers(),
                                json={"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5, "stream": True}) as resp:
                                async for line in resp.aiter_lines():
                                    if line.startswith("data: ") and line != "data: [DONE]":
                                        try:
                                            data = json.loads(line[6:])
                                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                            if content:
                                                yield sse({"type": "token", "text": content})
                                        except:
                                            pass
                        yield sse({"type": "done"})
                    except Exception as e:
                        yield sse({"type": "error", "message": str(e)})
                return StreamingResponse(event_generator(), media_type="text/event-stream",
                                          headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5})
                return {"reasoning": r.json()["choices"][0]["message"]["content"]}

        # ------------------------- CHAT (DEFAULT) -------------------------
        else:
            user_memory_str = await fetch_user_memory(user_id)
            async def event_generator():
                full_reply = ""
                try:
                    yield sse({"type": "status", "status": "thinking"})
                    messages = []
                    base_system = system_prompt or "You are HeloXAI, a helpful AI assistant."
                    if user_memory_str:
                        base_system += f"\n\n{user_memory_str}\n\nUse this context to personalize if relevant."
                    messages.append({"role": "system", "content": base_system})
                    if context:
                        messages.append({"role": "system", "content": f"Context: {context}"})
                    if documents:
                        doc_texts = [await extract_document_text(d) for d in documents]
                        messages.append({"role": "system", "content": f"Documents:\n{chr(10).join(doc_texts)}"})
                    messages.extend(history_messages)
                    messages.append({"role": "user", "content": prompt})
                    
                    reply = await chat_with_tools(user_id, messages)
                    full_reply = reply
                    for char in reply:
                        yield sse({"type": "token", "text": char})
                        await asyncio.sleep(0.005)
                    
                    await asyncio.to_thread(
                        lambda: supabase.table("messages").insert({
                            "id": str(uuid.uuid4()), "conversation_id": conversation_id,
                            "user_id": user_id, "role": "assistant", "content": reply,
                            "created_at": datetime.utcnow().isoformat()
                        }).execute()
                    )
                    yield sse({"type": "done"})
                except Exception as e:
                    logger.error(f"Chat failed: {e}")
                    yield sse({"type": "error", "message": str(e)})
                finally:
                    if full_reply:
                        asyncio.create_task(extract_and_save_memory(user_id, prompt, full_reply))
            return StreamingResponse(event_generator(), media_type="text/event-stream",
                                      headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/ask/universal failed: {e}")
        raise HTTPException(500, str(e))

# ---------- TTS/STT Endpoints ----------
@app.post("/tts")
async def text_to_speech(request: Request, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    text = re.sub(r"[#*`\[\]]", "", body.get("text", "")).strip()
    voice = body.get("voice", "alloy")
    if not text:
        raise HTTPException(400, "Text is required")
    
    if ELEVENLABS_API_KEY:
        try:
            voice_id = get_elevenlabs_voice_id(voice)
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                    json={"text": text, "model_id": "eleven_multilingual_v2"})
                return {"audio": base64.b64encode(r.content).decode(), "provider": "elevenlabs"}
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}")
    
    if OPENAI_API_KEY:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "tts-1", "voice": voice, "input": text})
            return {"audio": base64.b64encode(r.content).decode(), "provider": "openai"}
    raise HTTPException(500, "No TTS provider available")

@app.post("/stt")
async def speech_to_text(request: Request, current_user: dict = Depends(get_current_user_optional)):
    content_type = request.headers.get("content-type", "")
    audio_bytes = None
    
    if "multipart/form-data" in content_type:
        form = await request.form()
        audio_file = form.get("audio") or form.get("file")
        if not audio_file:
            raise HTTPException(400, "No audio file provided")
        audio_bytes = await audio_file.read()
    else:
        body = await request.json()
        audio_b64 = body.get("audio") or body.get("data")
        if not audio_b64:
            raise HTTPException(400, "No audio data provided")
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(audio_b64)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    
    try:
        if OPENAI_API_KEY:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(temp_path, "rb") as f:
                    r = await client.post("https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        files={"file": ("audio.wav", f)}, data={"model": "whisper-1"})
                    return {"text": r.json()["text"], "provider": "openai"}
        raise HTTPException(500, "No STT provider available")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/tts/voices")
async def get_tts_voices():
    return {"voices": {"openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                       "elevenlabs": ["rachel", "drew", "bella", "antoni", "josh", "grace"]},
            "providers": {"openai": bool(OPENAI_API_KEY), "elevenlabs": bool(ELEVENLABS_API_KEY)}}

# ---------- Chat Management ----------
@app.post("/stop")
async def stop_generation(request: Request, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    stream_id = body.get("stream_id")
    conversation_id = body.get("conversation_id")
    if stream_id:
        await stop_stream(stream_id)
        return {"status": "stopped", "stream_id": stream_id}
    if conversation_id:
        stopped = 0
        async with active_stream_lock:
            for sid in list(active_streams.keys()):
                if conversation_id in sid:
                    active_streams[sid].set()
                    stopped += 1
        return {"status": "stopped", "streams_stopped": stopped}
    async with active_stream_lock:
        for event in active_streams.values():
            event.set()
        stopped = len(active_streams)
        active_streams.clear()
    return {"status": "stopped", "streams_stopped": stopped}

@app.post("/regenerate")
async def regenerate_response(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    conversation_id = body.get("conversation_id")
    if not conversation_id:
        raise HTTPException(400, "conversation_id required")
    
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id") or str(uuid.uuid4())
    
    messages_res = await asyncio.to_thread(
        lambda: supabase.table("messages").select("id, role, content")
        .eq("conversation_id", conversation_id).eq("user_id", user_id)
        .order("created_at", desc=True).limit(10).execute())
    messages = messages_res.data or []
    
    if not messages:
        raise HTTPException(404, "No messages found")
    
    last_user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    last_assistant_id = next((m["id"] for m in messages if m["role"] == "assistant"), None)
    
    if not last_user_msg:
        raise HTTPException(400, "No user message found")
    
    history = await asyncio.to_thread(
        lambda: supabase.table("messages").select("role, content")
        .eq("conversation_id", conversation_id).eq("user_id", user_id)
        .order("created_at").limit(20).execute())
    history_messages = history.data or []
    if history_messages and history_messages[-1]["role"] == "assistant":
        history_messages = history_messages[:-1]
    
    if last_assistant_id:
        await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("id", last_assistant_id).execute())
    
    async def event_generator():
        try:
            yield sse({"type": "status", "status": "regenerating"})
            messages_payload = [{"role": "system", "content": "You are HeloXAI, a helpful AI assistant."}] + history_messages
            reply = await chat_with_tools(user_id, messages_payload)
            for char in reply:
                yield sse({"type": "token", "text": char})
                await asyncio.sleep(0.005)
            await asyncio.to_thread(
                lambda: supabase.table("messages").insert({
                    "id": str(uuid.uuid4()), "conversation_id": conversation_id,
                    "user_id": user_id, "role": "assistant", "content": reply,
                    "created_at": datetime.utcnow().isoformat()
                }).execute())
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

@app.post("/newchat")
async def create_new_chat(request: Request, response: Response, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id") or str(uuid.uuid4())
    if not (current_user or {}).get("id") and not request.cookies.get("guest_id"):
        response.set_cookie(key="guest_id", value=user_id, httponly=True, secure=True, samesite="lax", max_age=60*60*24*7)
    
    new_id = str(uuid.uuid4())
    await asyncio.to_thread(
        lambda: supabase.table("conversations").insert({
            "id": new_id, "user_id": user_id, "created_at": datetime.utcnow().isoformat()
        }).execute())
    return {"conversation_id": new_id, "user_id": user_id, "created": True}

@app.delete("/chat/{conversation_id}")
async def delete_chat(conversation_id: str, request: Request, current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id")
    if not user_id:
        raise HTTPException(401, "Auth required")
    await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("conversation_id", conversation_id).eq("user_id", user_id).execute())
    await asyncio.to_thread(lambda: supabase.table("conversations").delete().eq("id", conversation_id).eq("user_id", user_id).execute())
    await stop_stream(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}

@app.get("/chats")
async def list_chats(request: Request, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0),
                     current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id")
    if not user_id:
        return {"chats": [], "total": 0}
    conv_res = await asyncio.to_thread(
        lambda: supabase.table("conversations").select("id, title, created_at, updated_at")
        .eq("user_id", user_id).order("updated_at", desc=True).range(offset, offset + limit - 1).execute())
    return {"chats": conv_res.data or [], "total": len(conv_res.data or [])}

@app.get("/chat/{conversation_id}")
async def get_chat(conversation_id: str, request: Request, limit: int = Query(50, ge=1, le=200),
                   current_user: dict = Depends(get_current_user_optional)):
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id")
    if not user_id:
        raise HTTPException(401, "Auth required")
    conv_res = await asyncio.to_thread(
        lambda: supabase.table("conversations").select("id, title, created_at, system_prompt")
        .eq("id", conversation_id).eq("user_id", user_id).execute())
    if not conv_res.data:
        raise HTTPException(404, "Conversation not found")
    msg_res = await asyncio.to_thread(
        lambda: supabase.table("messages").select("id, role, content, created_at")
        .eq("conversation_id", conversation_id).order("created_at").range(0, limit - 1).execute())
    return {"conversation": conv_res.data[0], "messages": msg_res.data or []}

@app.patch("/chat/{conversation_id}")
async def update_chat(conversation_id: str, request: Request, current_user: dict = Depends(get_current_user_optional)):
    body = await request.json()
    user_id = (current_user or {}).get("id") or request.cookies.get("guest_id")
    if not user_id:
        raise HTTPException(401, "Auth required")
    update_data = {"updated_at": datetime.utcnow().isoformat()}
    if "title" in body:
        update_data["title"] = body["title"]
    if "system_prompt" in body:
        update_data["system_prompt"] = body["system_prompt"]
    await asyncio.to_thread(
        lambda: supabase.table("conversations").update(update_data).eq("id", conversation_id).eq("user_id", user_id).execute())
    return {"status": "updated", "conversation_id": conversation_id}

# ---------- Health & Info ----------
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/info")
async def info():
    return {"creator": CREATOR_INFO, "models": {"chat": CHAT_MODEL, "code": CODE_MODEL},
            "providers": {"groq": bool(GROQ_API_KEY), "openai": bool(OPENAI_API_KEY),
                          "elevenlabs": bool(ELEVENLABS_API_KEY), "judge0": bool(JUDGE0_KEY)}}

@app.get("/capabilities")
async def capabilities():
    return {"intents": ["chat", "code_generation", "code_execution", "math_calculation", "web_search",
                        "text_to_speech", "audio_transcription", "translation", "summarization", "reasoning",
                        "joke", "creative_writing", "vision_analysis", "image_generation"],
            "features": ["streaming", "conversation_history", "user_memory", "stop_generation", "regenerate"]}

if __name__ == "__main__":
    import uvicorn
    print(f"""
    ╔══════════════════════════════════════════╗
    ║         HeloXAI Server Started           ║
    ╠══════════════════════════════════════════╣
    ║  API: http://localhost:8000              ║
    ║  Docs: http://localhost:8000/docs        ║
    ╚══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
