# 🎬 AI Video Assistant

Turn a YouTube video or a local audio/video recording into meeting notes, then ask questions about it.

Give it a YouTube URL or a file path. It produces:

- **Title**: a short title for the session
- **Summary**: a bullet-point summary
- **Action items**: each with a task, an owner and a deadline
- **Key decisions**
- **Open questions**: unresolved topics that need follow-up
- **Chat**: question answering over the transcript (RAG)

It works with English audio and with Hinglish (mixed Hindi and English) audio.

---

## How it works

```
YouTube URL / local file
        │
        ▼
 ┌──────────────────┐   yt-dlp download → FFmpeg → 16 kHz mono WAV → 10-min chunks
 │  Audio processing│   (utils/audio_processor.py)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   english  → OpenAI Whisper (runs locally)
 │  Transcription   │   hinglish → Sarvam AI speech-to-text-translate (outputs English)
 └────────┬─────────┘   (core/transcriber.py)
          ▼
 ┌──────────────────┐   Google Gemini through LangChain (LCEL)
 │ Title / Summary /│   (core/summarizer.py, core/extractor.py)
 │   Extraction     │
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   Transcript split into 500-char chunks → all-MiniLM-L6-v2 embeddings
 │   RAG engine     │   → Chroma (./vector_db) → top-4 retrieval → Gemini answer
 └──────────────────┘   (core/vector_store.py, core/rag_engine.py)
```

## Project structure

```
.
├── app.py                  # Streamlit web UI
├── main.py                 # Command-line version of the pipeline, with interactive chat
├── test.py                 # Script that runs the pipeline on a hard-coded URL
├── core/
│   ├── transcriber.py      # Whisper / Sarvam transcription
│   ├── summarizer.py       # Map-reduce summary and title generation
│   ├── extractor.py        # Action items, decisions and questions
│   ├── vector_store.py     # Chroma vector store and HuggingFace embeddings
│   └── rag_engine.py       # LCEL RAG chain and ask_question()
├── utils/
│   └── audio_processor.py  # Download, WAV conversion and chunking
├── Requirements.txt
└── .env.example
```

## Requirements

- Python 3.10 or newer
- A **Google Gemini API key**
- A **Sarvam AI API key**, only if you want to use the Hinglish mode
- **FFmpeg**. If it is not on your `PATH`, the app uses the binary bundled with `imageio-ffmpeg`, which is installed from the requirements file.

Whisper runs on your own machine. The first run downloads the Whisper model and the embedding model. On a CPU, transcription takes a few minutes for every 10 minutes of audio.

## Setup

```bash
git clone <repo-url>
cd "AI-video assistant"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r Requirements.txt
```

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

| Variable           | Default                  | Description                                                     |
|--------------------|--------------------------|-----------------------------------------------------------------|
| `GEMINI_API_KEY`   | none (required)          | Google Gemini API key                                           |
| `GEMINI_MODEL`     | `gemini-3.5-flash-lite`  | Gemini model used for summaries, extraction and chat            |
| `WHISPER_MODEL`    | `small`                  | Whisper model size: `tiny`, `base`, `small`, `medium` or `large` |
| `SARVAM_API_KEY`   | none                     | Needed only when the language is `hinglish`                     |
| `SARVAM_STT_MODEL` | `saaras:v2.5`            | Sarvam speech-to-text-translate model                           |

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

1. In the sidebar, paste a YouTube URL or the path to a local file (`.mp4`, `.mp3`, `.wav`, `.webm` and so on).
2. Choose **english** or **hinglish**.
3. Click **⚡ Analyse**. The sidebar shows the progress of each pipeline step.
4. Read the title, summary, action items, decisions and questions. You can also open the full transcript.
5. Ask questions in **💬 Chat with your Meeting**.

### Command line

```bash
python main.py
```

The script asks for a source and a language. It prints the results and then starts a chat loop. Type `exit` to quit.

### From Python

```python
from dotenv import load_dotenv
load_dotenv()

from main import run_pipeline
from core.rag_engine import ask_question

result = run_pipeline("https://www.youtube.com/watch?v=...", language="english")
print(result["summary"])
print(ask_question(result["rag_chain"], "What were the main decisions?"))
```

## Notes

- Downloaded audio and the converted WAV chunks are saved in `downloades/`. Both `downloades/` and `vector_db/` are listed in `.gitignore`.
- The vector store holds one transcript at a time. Each new analysis deletes the previous collection, so the chat only answers from the current recording.
- Sarvam's synchronous API accepts at most 30 seconds of audio per request. In Hinglish mode, each chunk is split into 25-second pieces before it is sent.
- The summarizer splits the transcript into 200k-character chunks because Gemini has a large context window. Most recordings need only one or two API calls.

## Tech stack

Streamlit · LangChain (LCEL) · Google Gemini · OpenAI Whisper · Sarvam AI · ChromaDB · Sentence-Transformers · yt-dlp · FFmpeg / pydub
