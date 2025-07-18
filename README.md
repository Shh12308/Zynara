# 🤖 Zynara AI API — Full Stack Intelligence

Welcome to **Zynara**, a powerful full-stack AI API designed to handle a wide variety of tasks across natural language, vision, voice, translation, and knowledge domains.

This Space provides a RESTful API powered by **FastAPI**, integrating advanced open-source models and services like:
- 🧠 Mixtral (LLM)
- 👁️ BakLLaVA (image understanding)
- 🗣️ Whisper (speech-to-text)
- 🌍 NLLB-200 (translation)
- 🔎 DuckDuckGo Search
- ☁️ OpenWeather API
- ⚡ Wolfram Alpha
- ✅ Sightengine moderation
- 💾 Supabase memory and logging

---

## 🔧 Endpoints

### `/chat`  
Generate responses using Mixtral LLM  
**POST**  
```json
{
  "prompt": "Tell me a joke.",
  "user_id": "anonymous",
  "stream": false
}