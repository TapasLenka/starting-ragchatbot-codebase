# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

Requires Python 3.13+ and `uv`. On Windows, use Git Bash.

```bash
# Install dependencies
uv sync

# Start the server (runs from backend/ automatically)
./run.sh

# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000`. FastAPI auto-docs are at `http://localhost:8000/docs`.

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=<key>` (see `.env.example`).

There are no tests in this codebase.

## Common Operations

**Add new course documents**: Drop `.txt`, `.pdf`, or `.docx` files into `docs/` following the document format below, then restart the server. Duplicates (matched by course title) are skipped automatically.

**Reset the vector store**: Delete `backend/chroma_db/` and restart the server to force full re-ingestion.

```bash
rm -rf backend/chroma_db && ./run.sh
```

## Architecture

This is a full-stack RAG chatbot. The FastAPI server in `backend/` serves both the API and the static frontend from `frontend/`. All backend modules are imported without a package prefix (e.g., `from config import config`) because the server runs with `backend/` as the working directory.

### Request pipeline

A user query goes through this chain:

```
frontend/script.js  →  POST /api/query  →  app.py
  →  RAGSystem.query()          (rag_system.py)
    →  SessionManager            (session_manager.py)   — fetch/save conversation history
    →  AIGenerator               (ai_generator.py)      — Claude API call #1 (tool_choice: auto)
      →  CourseSearchTool        (search_tools.py)      — triggered if Claude picks "tool_use"
        →  VectorStore.search()  (vector_store.py)      — ChromaDB semantic search
      →  AIGenerator             (ai_generator.py)      — Claude API call #2 (synthesize answer)
```

The frontend (`frontend/script.js`) sends `{ query, session_id }` and receives `{ answer, sources, session_id }`. It renders the answer via `marked.parse()` (markdown) and shows sources in a collapsible `<details>` element.

### Key design decisions

**Two ChromaDB collections**: `course_catalog` stores one document per course (title + metadata) used for fuzzy course name resolution. `course_content` stores all text chunks and is what gets queried for answers.

**Tool-based retrieval**: Claude decides whether to call `search_course_content`. General-knowledge questions skip ChromaDB entirely. The tool accepts optional `course_name` and `lesson_number` filters; `course_name` is resolved semantically against `course_catalog` before filtering `course_content`.

**Two-call Claude pattern**: Call #1 includes tools and may return `stop_reason="tool_use"`. If so, `_handle_tool_execution()` runs the tool, appends the `tool_result` message, and makes Call #2 without tools to get the final answer.

**Conversation history** is injected into the system prompt (not as separate messages). `SessionManager` keeps the last `MAX_HISTORY=2` exchanges (4 messages total) per in-memory session. Sessions are lost on server restart.

**Document ingestion** happens at startup (`startup_event` in `app.py`): all `.txt/.pdf/.docx` files in `docs/` are processed. Duplicate courses (matched by title) are skipped. `DocumentProcessor` parses a specific file format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
<content...>

Lesson 2: <title>
...
```
Text is split into sentence-aware chunks (800 chars, 100-char overlap). The first chunk of each lesson is prefixed with `"Lesson N content: "` for context.

### Gotchas

- `main.py` in the project root is an unused stub — the real entry point is `run.sh` / uvicorn in `backend/`.
- `DevStaticFiles` is defined in `app.py` but never used; the actual mount at the bottom of `app.py` uses the base `StaticFiles` class.
- Document chunking has an inconsistency: all lessons except the last are prefixed `"Lesson N content: ..."`, while the last lesson uses `"Course {title} Lesson N content: ..."` — both end up in `course_content` and are searchable the same way.
- All backend imports are flat (e.g. `from config import config`) because uvicorn runs with `backend/` as the working directory. Do not add a package prefix.

### Configuration (backend/config.py)

| Setting | Default | Notes |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Hard-coded, not in `.env` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers |
| `CHUNK_SIZE` | 800 | characters |
| `CHUNK_OVERLAP` | 100 | characters |
| `MAX_RESULTS` | 5 | ChromaDB top-k |
| `MAX_HISTORY` | 2 | exchanges remembered |
| `CHROMA_PATH` | `./chroma_db` | relative to `backend/` |
