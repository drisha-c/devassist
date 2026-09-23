from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Create the Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "DevAssist API is running"}

@app.post("/ask")
def ask(payload: Question):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=payload.question
    )
    return {"question": payload.question, "answer": response.text}