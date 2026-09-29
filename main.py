import time
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


def get_index():
    if "chunks" not in _index:
        chunks = rag.load_chunks()
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