import os
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

sentences = [
    "How do I reset my password?",
    "Steps to recover a forgotten login credential",
    "The best pizza toppings for a party",
]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
)
vectors = [np.array(e.values) for e in result.embeddings]

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("Numbers per vector:", len(vectors[0]))
print("password vs recover login:", cosine(vectors[0], vectors[1]))
print("password vs pizza:        ", cosine(vectors[0], vectors[2]))