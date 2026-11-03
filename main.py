import time
import uuid
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from google import genai
from google.genai import errors
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import secrets

import rag

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("DEVASSIST_API_KEY")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_index = {}
_pending_fixes = {}  # in-memory store: fix_id -> suggestion details
_request_log = []  # in-memory list of every /ask-docs call, for /stats


def get_index():
    if "chunks" not in _index:
        chunks = rag.load_all_chunks()
        vectors = rag.embed_texts(
            [c["title"] + "\n" + c["text"] for c in chunks]
        )
        _index["chunks"] = chunks
        _index["vectors"] = vectors
    return _index["chunks"], _index["vectors"]


def check_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: DEVASSIST_API_KEY is not set",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "DevAssist API is running"}


@app.post("/ask")
@limiter.limit("5/minute")
def ask(request: Request, payload: Question, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=payload.question
            )
            return {"question": payload.question, "answer": response.text}

        except errors.ServerError:
            if attempt < max_attempts:
                time.sleep(2 * attempt)
            else:
                raise HTTPException(
                    status_code=503,
                    detail="The AI model is busy right now. Please try again in a minute."
                )
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Daily quota reached for the AI model. Please try again tomorrow.",
                )
            raise HTTPException(
                status_code=502,
                detail="The AI model rejected the request. Check the server's API key and model settings.",
            )


@app.post("/ask-docs")
@limiter.limit("5/minute")
def ask_docs(request: Request, payload: Question, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    chunks, vectors = get_index()
    text, sources, status, top_score, elapsed_ms = rag.answer(
        payload.question, chunks, vectors
    )

    _request_log.append({
        "question": payload.question,
        "status": status,
        "top_score": top_score,
        "elapsed_ms": elapsed_ms,
        "grounded": len(sources) > 0,
    })

    body = {
        "question": payload.question,
        "answer": text,
        "sources": sources,
    }
    if status != 200:
        raise HTTPException(status_code=status, detail=body)
    return body


@app.get("/stats")
def stats():
    if not _request_log:
        return {"total_requests": 0, "message": "No requests logged yet"}

    total = len(_request_log)
    grounded = sum(1 for r in _request_log if r["grounded"])
    avg_score = sum(r["top_score"] for r in _request_log) / total
    avg_latency = sum(r["elapsed_ms"] for r in _request_log) / total

    return {
        "total_requests": total,
        "grounded_answers": grounded,
        "unanswered": total - grounded,
        "average_top_score": round(avg_score, 3),
        "average_latency_ms": round(avg_latency),
        "recent_requests": _request_log[-5:],
    }


@app.post("/suggest-fix")
@limiter.limit("5/minute")
def suggest_fix_endpoint(request: Request, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    chunks, vectors = get_index()
    result = rag.suggest_fix(chunks, vectors)

    if result[0] is None:
        _, message, sources, status = result
        return {"error_found": False, "message": message}

    error_line, suggestion, sources, status = result
    if status != 200:
        raise HTTPException(status_code=status, detail=suggestion)

    fix_id = str(uuid.uuid4())[:8]
    _pending_fixes[fix_id] = {
        "error": error_line,
        "suggestion": suggestion,
        "sources": sources,
        "approved": False,
    }

    return {
        "fix_id": fix_id,
        "error_found": True,
        "error": error_line,
        "suggestion": suggestion,
        "sources": sources,
        "status": "awaiting_approval",
    }


@app.post("/approve-fix/{fix_id}")
@limiter.limit("5/minute")
def approve_fix(fix_id: str, request: Request, x_api_key: str = Header(None)):
    check_api_key(x_api_key)

    fix = _pending_fixes.get(fix_id)
    if fix is None:
        raise HTTPException(status_code=404, detail="No pending fix with that ID")

    if fix["approved"]:
        return {"fix_id": fix_id, "status": "already_approved"}

    fix["approved"] = True
    action_taken = f"[SIMULATED] Would execute: {fix['suggestion']}"

    return {
        "fix_id": fix_id,
        "status": "approved_and_executed",
        "action_taken": action_taken,
    }