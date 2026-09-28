import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import errors
from dotenv import load_dotenv
import os

import rag

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

app = FastAPI()

# Cache for the document index (built on first use, then reused)
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


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "DevAssist API is running"}


@app.post("/ask")
def ask(payload: Question):
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
def ask_docs(payload: Question):
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