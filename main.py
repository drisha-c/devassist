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
    if x_api_key != API_KEY:
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


@app.post("/ask-docs")
@limiter.limit("5/minute")
def ask_docs(request: Request, payload: Question, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    chunks, vectors = get_index()
    text, sources, status = rag.answer(payload.question, chunks, vectors)
    body = {
        "question": payload.question,
        "answer": text,
        "sources": sources,
    }
    if status != 200:
        raise HTTPException(status_code=status, detail=body)
    return body


@app.post("/suggest-fix")
@limiter.limit("5/minute")
def suggest_fix_endpoint(request: Request, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    chunks, vectors = get_index()
    result = rag.suggest_fix(chunks, vectors)

    if result[0] is None:
        # no error found
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
    # This is a mocked action. A real system would call infrastructure
    # APIs here (restart a service, roll back a deploy, etc.).
    action_taken = f"[SIMULATED] Would execute: {fix['suggestion']}"

    return {
        "fix_id": fix_id,
        "status": "approved_and_executed",
        "action_taken": action_taken,
    }