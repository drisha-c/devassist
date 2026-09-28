import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import errors
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

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
                model="gemini-3.6-flash",
                contents=payload.question
            )
            return {"question": payload.question, "answer": response.text}

        except errors.ServerError:
            # Google-side problem (like 503). Wait and retry.
            if attempt < max_attempts:
                time.sleep(2 * attempt)  # waits 2s, then 4s
            else:
                raise HTTPException(
                    status_code=503,
                    detail="The AI model is busy right now. Please try again in a minute."
                )