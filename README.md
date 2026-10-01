# DevAssist

A RAG-powered DevOps copilot that answers questions grounded in a project's own documentation and logs, and demonstrates a human-in-the-loop guarded action pattern for suggesting and approving fixes.

Live API: https://devassist-1.onrender.com/docs

## What it does

- **`/ask`** - plain LLM question answering (Gemini), no grounding
- **`/ask-docs`** - RAG: answers only from this project's own docs and logs, with cited sources, and honestly says "I don't know" when nothing relevant is found
- **`/suggest-fix`** - scans logs for the most recent error, retrieves relevant context, and drafts a grounded remediation suggestion
- **`/approve-fix/{fix_id}`** - requires explicit human approval before "executing" (simulated) the suggested fix
- **`/stats`** - live observability: total requests, retrieval score averages, latency, and how many answers were grounded vs unanswered

## Why it's built this way

**Two data sources, two chunking strategies.** Markdown docs are chunked by heading (natural semantic boundaries). Logs are chunked by fixed-size line windows (no natural headings exist). Both are embedded and searched together, with results tagged by source type so the answer's grounding is visible.

**A relevance cutoff, not just top-k.** Retrieval returns the top-k closest chunks by cosine similarity, but anything below a similarity threshold is discarded before being shown to the model. This is what lets the system say "I don't know" instead of generating a plausible-sounding but ungrounded answer. Testing showed clear separation: real, answerable questions scored 0.65+, while an unrelated control question ("what is the capital of France") scored 0.47 and was correctly rejected.

**Known limitation, documented honestly.** Testing found that natural-language questions about log content retrieve well (e.g. "was there a rate limit error?"), but exact-value lookups like timestamps often fail, since embeddings capture semantic meaning, not literal values. A production system would likely pair semantic search with structured/keyword filtering for this kind of query. (Full note in `docs/notes.md`.)

**Human-in-the-loop by design.** `/suggest-fix` never executes anything. It proposes an action and stores it under an ID. Only `/approve-fix/{id}`, a separate, explicit call, triggers the (simulated) action. Double-approval and invalid IDs are both handled correctly.

## Stack

- **FastAPI** - API framework
- **Google Gemini** (`gemini-3.5-flash-lite` for generation, `gemini-embedding-001` for embeddings) - chosen for a generous free-tier daily quota after hitting rate limits during development
- **NumPy** - cosine similarity search (no vector DB; small enough corpus that an in-memory index is appropriate)
- **slowapi** - rate limiting (5 requests/minute per IP per endpoint)
- **Render** - deployment, auto-redeploys on push to `main`
- **GitHub Actions** - CI, runs a pytest suite (chunking/parsing logic only, no live API calls) on every push

## Running locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# create a .env file with:
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-3.5-flash-lite
# DEVASSIST_API_KEY=your_own_chosen_key

uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Running tests

```bash
pytest tests/ -v
```

Tests cover chunking and parsing logic only, no live Gemini calls, so they run free and fast (~1 second) in CI.

## What I'd build next

- Vector database (pgvector or similar) instead of in-memory NumPy, for scale
- Keyword/structured search alongside semantic search, to fix the timestamp-lookup limitation
- Real action execution behind the approval flow, with actual infrastructure API calls
- Persistent storage for `/stats` and pending fixes (currently in-memory, resets on restart)
