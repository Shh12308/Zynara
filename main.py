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
from enum import Enum
from dataclasses import dataclass
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
    legacy_intent: str  # For backward compatibility

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
        self.negation_words = {
            "don't", "dont", "do not", "doesn't", "doesnt", "does not",
            "didn't", "didnt", "did not", "never", "no", "not", "without",
            "skip", "avoid", "except", "but not", "ignore", "rather than"
        }
        
        # Map new intents to legacy intent names for backward compatibility
        self.legacy_intent_map = {
            IntentCategory.IMAGE_GENERATION: "image_generation",
            IntentCategory.VIDEO_GENERATION: "video_generation",
            IntentCategory.AUDIO_GENERATION: "audio_generation",
            IntentCategory.CODE_GENERATION: "code_generation",
            IntentCategory.CODE_REVIEW: "code_generation",
            IntentCategory.CODE_DEBUG: "code_generation",
            IntentCategory.CODE_EXECUTION: "code_execution",
            IntentCategory.DOCUMENT_CREATION: "document_creation",
            IntentCategory.DOCUMENT_ANALYSIS: "document_creation",
            IntentCategory.DATA_ANALYSIS: "data_analysis",
            IntentCategory.DATA_VISUALIZATION: "data_analysis",
            IntentCategory.WEB_DEVELOPMENT: "code_generation",
            IntentCategory.API_DEVELOPMENT: "code_generation",
            IntentCategory.DATABASE: "code_generation",
            IntentCategory.TRANSLATION: "translation",
            IntentCategory.SUMMARIZATION: "summarization",
            IntentCategory.EXPLANATION: "reasoning",
            IntentCategory.REASONING: "reasoning",
            IntentCategory.CREATIVE_WRITING: "creative_writing",
            IntentCategory.MATHEMATICAL: "math_calculation",
            IntentCategory.RESEARCH: "web_search",
            IntentCategory.WEB_SEARCH: "web_search",
            IntentCategory.TEXT_TO_SPEECH: "text_to_speech",
            IntentCategory.AUDIO_TRANSCRIPTION: "audio_transcription",
            IntentCategory.VISION_ANALYSIS: "vision_analysis",
            IntentCategory.JOKE: "joke",
            IntentCategory.CONVERSATION: "chat",
        }

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        self.patterns = {
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)',
                r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)',
                r'\b(text\s+to\s+image|txt2img|img2img)',
                r'\b(visualize|visualise)\s+(this|that|the|it)',
                r'\b(dall[eé]|midjourney|stable\s+diffusion|sd\s*xl|flux)',
                r'\b(generate|create)\s+(some\s+)?art',
                r'\bmake\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster)',
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
                r'\bimplement\s+(the|a|this)\s+(\w+\s+)?(pattern|algorithm|logic|feature)',
                r'\b(python|javascript|java|typescript)\s+code',
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
            IntentCategory.CODE_EXECUTION: [
                r'\b(run|execute)\s+(this|my|the)\s+(code|script|program)',
                r'\b(output|result)\s+(of|for)\s+(this|the)\s+(code|program)',
                r'\b(test\s+(this|my|the)\s+(code|function|program))',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
                r'\b(format\s+(as|this\s+as|it\s+as)\s+(a\s+)?(pdf|document|report|letter|markdown))',
                r'\b(professional|formal|business)\s+(document|letter|email|report)',
            ],
            IntentCategory.DOCUMENT_ANALYSIS: [
                r'\b(analyze|analysis|analyse|read|parse|extract)\s+(this|the|my)\s+(document|pdf|file|doc|report)',
                r'\b(summarize\s+(this|the)\s+(document|pdf|file|report))',
                r'\b(what(\'s|\s+is)\s+(in|inside)\s+(this|the)\s+(document|pdf|file))',
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
                r'\bapi\s*(endpoint|route|handler|controller|gateway)',
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
                r'\b(in\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu))',
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
                r'\b(definition|meaning)\s*(of|for)\s+',
                r'\b(understand(ing)?)\s*(this|how|why|what|better)?',
                r'\b(break\s+down|simplify|elaborate)\s+',
            ],
            IntentCategory.REASONING: [
                r'\b(reason|think)\s+(through|step\s+by\s+step)',
                r'\b(step\s+by\s+step)',
                r'\b(logical|critical|analytical)\s+(thinking|reasoning|analysis)',
                r'\b(think\s+(about|carefully|through))',
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
                r'\b(integral|derivative|differentiat|integrat)\s*(of|the)?',
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
            IntentCategory.WEB_SEARCH: [
                r'\b(search|google)\s+(for|about|on)\s+',
                r'\b(what\s+is|who\s+is|where\s+is|when\s+(is|was|did))\s+',
                r'\b(latest|current|recent|news|update[s]?)\s+(on|about|for)\s+',
                r'\b(find\s+(me|out))\s+',
            ],
            IntentCategory.TEXT_TO_SPEECH: [
                r'\b(speak|say|read|narrate|pronounce)\s+(this|it|the|that)\s+',
                r'\b(text\s+to\s+speech|tts)\b',
                r'\b(read\s+(this|aloud|loud|to\s+me))',
            ],
            IntentCategory.AUDIO_TRANSCRIPTION: [
                r'\b(transcribe\s+(this\s+)?(audio|voice|recording|speech))',
                r'\b(speech\s+to\s+text|stt)\b',
                r'\b(convert\s+(this|the)\s+(audio|voice|speech)\s+to\s+text)',
            ],
            IntentCategory.VISION_ANALYSIS: [
                r'\b(analyze|describe|what(?:\'s| is)\s+in)\s+(this\s+)?(image|picture|photo)',
                r'\b(describe|explain|identify)\s+(this|the)\s+(image|picture|photo|visual)',
                r'\b(what\s+(do\s+you\s+)?see\s+in)',
                r'\b(ocr|text\s+recognition|read\s+(the\s+)?text\s+(from|in))',
            ],
            IntentCategory.JOKE: [
                r'\btell\s+(me\s+)?(a\s+)?joke',
                r'\bmake\s+me\s+laugh',
                r'\bsomething\s+funny',
                r'\b(joke|funny|humor|humour)\s+(about|on|for)\s+',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$',
                r'^(how\s+are\s+you|how(\'s|\s+is)\s+it\s+going|what(\'s|\s+is)\s+up)[\s!.?]*$',
                r'^(bye|goodbye|see\s+you|farewell)[\s!.?]*$',
                r'^(sure|okay|ok|got\s+it|understood)[\s!.?]*$',
            ],
        }

        # Compile all patterns
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.patterns.items()
        }

    def _init_synonyms(self):
        """Initialize synonym mappings for fuzzy keyword matching"""
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
        """Check if there's a negation word before the keyword (within 6 words)"""
        words_before = text[:keyword_pos].lower().split()[-6:]
        preceding_text = " ".join(words_before)
        return any(neg in preceding_text for neg in self.negation_words)

    def _calculate_confidence(
            self,
            matched_keywords: List[str],
            matched_patterns: List[str],
            text_length: int
    ) -> float:
        """Calculate confidence score based on matches"""
        if not matched_keywords and not matched_patterns:
            return 0.0

        pattern_confidence = min(len(matched_patterns) * 0.35, 0.65)
        keyword_confidence = min(len(matched_keywords) * 0.12, 0.25)
        multi_signal_bonus = 0.1 if (matched_keywords and matched_patterns) else 0.0
        length_factor = max(0.5, 1.0 - (text_length / 1500) * 0.4)

        confidence = (pattern_confidence + keyword_confidence + multi_signal_bonus) * length_factor
        return min(confidence, 1.0)

    def _are_related_intents(self, intent1: IntentCategory, intent2: IntentCategory) -> bool:
        """Check if two intents are related (can be sub-intents)"""
        related_groups = [
            {IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG, IntentCategory.CODE_EXECUTION},
            {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION},
            {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION},
            {IntentCategory.WEB_DEVELOPMENT, IntentCategory.API_DEVELOPMENT, IntentCategory.DATABASE},
            {IntentCategory.DOCUMENT_CREATION, IntentCategory.DOCUMENT_ANALYSIS, IntentCategory.RESEARCH},
            {IntentCategory.EXPLANATION, IntentCategory.SUMMARIZATION, IntentCategory.REASONING},
            {IntentCategory.RESEARCH, IntentCategory.WEB_SEARCH},
            {IntentCategory.TEXT_TO_SPEECH, IntentCategory.AUDIO_GENERATION},
            {IntentCategory.AUDIO_TRANSCRIPTION, IntentCategory.AUDIO_GENERATION},
        ]
        for group in related_groups:
            if intent1 in group and intent2 in group:
                return True
        return False

    def detect_intents(self, text: str, threshold: float = 0.25) -> List[IntentResult]:
        """Detect all intents with confidence scores"""
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
                    legacy_intent = self.legacy_intent_map.get(intent, "chat")
                    results.append(IntentResult(
                        intent=intent,
                        confidence=confidence,
                        sub_intents=[],
                        keywords_matched=matched_keywords,
                        patterns_matched=matched_patterns,
                        legacy_intent=legacy_intent
                    ))

        results.sort(key=lambda x: x.confidence, reverse=True)

        if results:
            primary = results[0]
            for result in results[1:]:
                if self._are_related_intents(primary.intent, result.intent):
                    primary.sub_intents.append(result.intent)

        return results[:1] if results else []

    def get_primary_intent(self, text: str) -> Optional[IntentResult]:
        """Get the highest confidence intent"""
        results = self.detect_intents(text)
        return results[0] if results else None

    def get_action_type(self, text: str) -> str:
        """Get high-level action type for routing (legacy compatible)"""
        intent = self.get_primary_intent(text)
        if not intent:
            return "chat"
        return intent.legacy_intent

    def get_required_tools(self, text: str) -> List[str]:
        """Determine which tools/APIs are needed"""
        intent = self.get_primary_intent(text)
        if not intent:
            return ["llm"]

        tool_map = {
            IntentCategory.IMAGE_GENERATION: ["image_gen", "llm"],
            IntentCategory.VIDEO_GENERATION: ["video_gen", "llm"],
            IntentCategory.AUDIO_GENERATION: ["audio_gen", "llm"],
            IntentCategory.CODE_GENERATION: ["code_gen", "llm"],
            IntentCategory.CODE_REVIEW: ["llm"],
            IntentCategory.CODE_DEBUG: ["code_exec", "llm"],
            IntentCategory.CODE_EXECUTION: ["code_exec", "llm"],
            IntentCategory.DOCUMENT_CREATION: ["doc_gen", "llm"],
            IntentCategory.DOCUMENT_ANALYSIS: ["doc_parser", "llm"],
            IntentCategory.DATA_ANALYSIS: ["code_exec", "data_processing", "llm"],
            IntentCategory.DATA_VISUALIZATION: ["code_exec", "llm"],
            IntentCategory.WEB_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.API_DEVELOPMENT: ["code_exec", "llm"],
            IntentCategory.DATABASE: ["database", "code_exec", "llm"],
            IntentCategory.TRANSLATION: ["llm"],
            IntentCategory.SUMMARIZATION: ["llm"],
            IntentCategory.EXPLANATION: ["llm"],
            IntentCategory.REASONING: ["llm"],
            IntentCategory.CREATIVE_WRITING: ["llm"],
            IntentCategory.MATHEMATICAL: ["wolfram", "code_exec", "llm"],
            IntentCategory.RESEARCH: ["web_search", "llm"],
            IntentCategory.WEB_SEARCH: ["web_search", "llm"],
            IntentCategory.TEXT_TO_SPEECH: ["tts", "llm"],
            IntentCategory.AUDIO_TRANSCRIPTION: ["stt"],
            IntentCategory.VISION_ANALYSIS: ["vision", "llm"],
            IntentCategory.JOKE: ["llm"],
            IntentCategory.CONVERSATION: ["llm"],
        }

        tools = list(tool_map.get(intent.intent, ["llm"]))

        for sub_intent in intent.sub_intents:
            for tool in tool_map.get(sub_intent, []):
                if tool not in tools:
                    tools.append(tool)

        return tools

    def get_code_system_prompt(self, text: str) -> str:
        """Get specialized system prompt based on code sub-intent"""
        intent = self.get_primary_intent(text)
        if not intent:
            return "Generate clean, well-documented code. Return ONLY code in a code block."

        sub_prompts = {
            IntentCategory.CODE_DEBUG: """You are an expert debugger. When analyzing code issues:
1. Identify the root cause of the bug/error
2. Explain WHY it's happening (not just what)
3. Provide the exact fix with clear code blocks
4. Suggest how to prevent similar issues
Be precise and practical.""",

            IntentCategory.CODE_REVIEW: """You are a senior code reviewer. Provide constructive feedback on:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Best practices and design patterns
5. Security concerns
Be specific and actionable.""",

            IntentCategory.CODE_GENERATION: "Generate clean, well-documented code. Return ONLY code in a code block.",

            IntentCategory.WEB_DEVELOPMENT: """You are a full-stack web developer expert. When building web components:
1. Use modern best practices and frameworks
2. Ensure responsive design
3. Consider accessibility (a11y)
4. Include proper styling
5. Make components reusable and maintainable
Provide complete, ready-to-use code.""",

            IntentCategory.API_DEVELOPMENT: """You are an API development expert. When creating APIs:
1. Follow RESTful principles (or GraphQL best practices)
2. Include proper error handling and status codes
3. Add input validation
4. Consider security (auth, rate limiting)
5. Document endpoints clearly
Provide complete, production-ready code.""",

            IntentCategory.DATABASE: """You are a database expert. When working with databases:
1. Design efficient, normalized schemas
2. Write optimized queries
3. Include proper indexes
4. Consider data integrity with constraints
5. Follow SQL best practices
Provide complete, ready-to-execute SQL/ORM code.""",
        }

        return sub_prompts.get(intent.intent, "Generate clean, well-documented code. Return ONLY code in a code block.")


# Singleton instance
_detector = None


def get_detector() -> AdvancedIntentDetector:
    global _detector
    if _detector is None:
        _detector = AdvancedIntentDetector()
    return _detector


def detect_intent(prompt: str) -> tuple:
    """Legacy-compatible intent detection function"""
    if not prompt:
        return ("chat", 0.0)
    
    result = get_detector().get_primary_intent(prompt)
    
    if result:
        return (result.legacy_intent, result.confidence)
    
    return ("chat", 0.0)


def detect_intent_advanced(prompt: str) -> Optional[IntentResult]:
    """Get detailed intent with confidence and metadata"""
    return get_detector().get_primary_intent(prompt)


def get_required_tools(prompt: str) -> List[str]:
    """Get list of tools needed for the request"""
    return get_detector().get_required_tools(prompt)


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


async def solve_math_with_reasoning(prompt: str, stream: bool) -> StreamingResponse:
    """Enhanced math solving with step-by-step reasoning"""
    system_prompt = """You are a mathematical expert. When solving math problems:
1. Show your work step-by-step
2. Explain each step clearly
3. Use proper mathematical notation
4. Verify your answer
5. If it's a proof, be rigorous

Format complex equations clearly using LaTeX-style notation where appropriate."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Solve step by step:\n\n{prompt}"}
    ]
    
    async def event_generator():
        try:
            yield sse({"type": "status", "status": "computing"})
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", GROQ_URL, headers=get_groq_headers(),
                    json={"model": CHAT_MODEL, "messages": messages, "max_tokens": 4096, "temperature": 0.3, "stream": True}) as resp:
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
        return {"id": token}
    return {}


# ---------- Intent Analysis Endpoint ----------
@app.post("/analyze-intent")
async def analyze_intent_endpoint(request: Request):
    """Analyze the intent of a prompt without executing it"""
    body = await request.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(400, "Prompt required")

    intent_result = detect_intent_advanced(prompt)
    required_tools = get_required_tools(prompt)
    legacy_intent, confidence = detect_intent(prompt)

    return {
        "intent": intent_result.to_dict() if intent_result else None,
        "legacy_intent": legacy_intent,
        "confidence": confidence,
        "required_tools": required_tools,
        "action_type": legacy_intent,
    }


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

        # ------------------------- ADVANCED INTENT DETECTION -------------------------
        intent_result = detect_intent_advanced(prompt)
        detected_intent = intent_result.legacy_intent if intent_result else "chat"
        confidence = intent_result.confidence if intent_result else 0.0
        required_tools = get_required_tools(prompt)
        
        logger.info(f"[INTENT] {detected_intent} (confidence: {confidence:.2%}) tools={required_tools} sub_intents={[i.value for i in (intent_result.sub_intents if intent_result else [])]}")
        
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
            if enable_cot or stream:
                return await solve_math_with_reasoning(prompt, stream)
            return await solve_math(prompt)

        # ------------------------- CODE GENERATION -------------------------
        elif detected_intent == "code_generation":
            lang = language or detect_language(prompt)
            code_system = get_detector().get_code_system_prompt(prompt)
            code_prompt = f"Write a {lang} program for: {prompt}"
            if context:
                code_prompt = f"Context: {context}\n\n{code_prompt}"
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                    "model": CODE_MODEL,
                    "messages": [{"role": "system", "content": code_system},
                                 {"role": "user", "content": code_prompt}],
                    "max_tokens": max_tokens, "temperature": temperature
                })
                return {"language": lang, "code": r.json()["choices"][0]["message"]["content"], "sub_intent": intent_result.intent.value if intent_result else None}

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
        elif detected_intent in ("web_search", "research"):
            query = re.sub(r"^(search for|look up|find|google|research)\s+", "", prompt.lower(), flags=re.IGNORECASE)
            if detected_intent == "research":
                query = re.sub(r"^(research|investigate|look into)\s+(about|on|into)?\s*", "", prompt.lower(), flags=re.IGNORECASE)
            
            research_system = """You are a research assistant. When answering research queries:
1. Provide well-structured, factual information
2. Cite sources when possible
3. Present multiple perspectives on controversial topics
4. Identify gaps in current knowledge
5. Suggest further areas of investigation""" if detected_intent == "research" else None
            
            if stream:
                async def event_generator():
                    try:
                        yield sse({"type": "status", "status": "searching"})
                        results = await duckduckgo_search(query)
                        yield sse({"type": "search_results", "results": results})
                        yield sse({"type": "status", "status": "summarizing"})
                        messages = [{"role": "user", "content": f"Answer: {query}\nResults: {json.dumps(results)}"}]
                        if research_system:
                            messages.insert(0, {"role": "system", "content": research_system})
                        async with httpx.AsyncClient(timeout=60) as client:
                            r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                                "model": CHAT_MODEL, "messages": messages,
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
            text = re.sub(r"^(speak|say|read|narrate)\s+(this|it|the|that)?\s*", "", text, flags=re.IGNORECASE)
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

        # ------------------------- CREATIVE WRITING -------------------------
        elif detected_intent == "creative_writing":
            creative_system = """You are a creative writing expert. When writing creative content:
1. Use vivid, engaging language
2. Create compelling characters and narratives
3. Pay attention to rhythm and flow
4. Match the requested style or genre
5. Be original and imaginative"""
            
            messages = [{"role": "system", "content": creative_system}] + history_messages[-6:] + [{"role": "user", "content": prompt}]
            
            if stream:
                async def event_generator():
                    try:
                        yield sse({"type": "status", "status": "creating"})
                        async with httpx.AsyncClient(timeout=120) as client:
                            async with client.stream("POST", GROQ_URL, headers=get_groq_headers(),
                                json={"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.9, "stream": True}) as resp:
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
                r = await client.post(GROQ_URL, headers=get_groq_headers(), json={
                    "model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.9
                })
                return {"creative_content": r.json()["choices"][0]["message"]["content"]}

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
                          "elevenlabs": bool(ELEVENLABS_API_KEY), "judge0": bool(JUDGE0_KEY)},
            "intent_detection": "advanced"}


@app.get("/capabilities")
async def capabilities():
    return {
        "intents": [intent.value for intent in IntentCategory],
        "intent_categories": {
            "generation": ["image_generation", "video_generation", "audio_generation", "code_generation", "document_creation", "creative_writing"],
            "analysis": ["code_review", "code_debug", "document_analysis", "data_analysis", "data_visualization", "vision_analysis"],
            "execution": ["code_execution"],
            "reasoning": ["mathematical", "reasoning", "explanation"],
            "search": ["web_search", "research"],
            "media": ["text_to_speech", "audio_transcription"],
            "language": ["translation"],
            "content": ["summarization"],
            "social": ["joke", "conversation"]
        },
        "features": [
            "streaming",
            "conversation_history",
            "user_memory",
            "stop_generation",
            "regenerate",
            "intent_detection",
            "confidence_scoring",
            "sub_intent_detection",
            "tool_requirement_analysis"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print(f"""
    ╔══════════════════════════════════════════╗
    ║         HeloXAI Server Started           ║
    ╠══════════════════════════════════════════╣
    ║  API: http://localhost:8000              ║
    ║  Docs: http://localhost:8000/docs        ║
    ║  Intent Detection: ADVANCED              ║
    ╚══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
