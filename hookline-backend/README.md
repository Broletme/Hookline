# Hookline Backend

FastAPI backend for the Hookline viral clip extraction pipeline.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in env vars
cp .env.example .env
# Edit .env with your GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

# 4. Run the Supabase schema migration
# Open schema.sql in the Supabase SQL editor and execute it once.

# 5. Start the dev server
uvicorn main:app --reload --port 8000
```

## Prerequisites

- **Python 3.11+**
- **ffmpeg** — must be on PATH (`ffmpeg -version` to verify)
- **yt-dlp** — installed as a Python package via requirements.txt
- **Groq API key** — https://console.groq.com
- **Supabase project** — https://supabase.com (free tier works)

## Pipeline

```
POST /jobs  →  download  →  transcribe  →  score  →  clip  →  upload  →  done
```

| Stage      | Module         | Description |
|------------|---------------|-------------|
| Download   | download.py    | yt-dlp fetches video; ffmpeg extracts 16 kHz mono WAV |
| Transcribe | transcribe.py  | Groq Whisper large-v3 returns word-level timestamps |
| Score      | scorer.py      | LLaMA 3.3 70B identifies 30–90 s viral candidates |
| Clip       | clipper.py     | ffmpeg cuts H.264/AAC MP4 clips |
| Upload     | storage.py     | Clips uploaded to Supabase Storage; job row updated |

## Environment Variables

| Variable                | Description |
|------------------------|-------------|
| `GROQ_API_KEY`         | Groq platform API key |
| `SUPABASE_URL`         | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Service-role secret key (bypasses RLS) |
| `SUPABASE_CLIPS_BUCKET`| Storage bucket name (default: `clips`) |
| `WORK_DIR`             | Local temp directory (default: `./workdir`) |
| `CORS_ORIGINS`         | Comma-separated allowed origins |

## API

```
POST /jobs          { youtube_url: string }  →  202 { id, status }
GET  /jobs/{id}                              →  200 { id, status, clips?, error? }
GET  /health                                 →  200 { status: "ok" }
```
