import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MIN_SCORE = 0.6  # chunks scoring below this are treated as not relevant


def load_chunks(folder="docs"):
    chunks = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("\\#", "#")

        # each section starts at a line beginning with "# "
        sections = text.split("\n# ")
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if section.startswith("# "):
                section = section[2:]

            title, _, body = section.partition("\n")
            chunks.append({
                "source": path.name,
                "title": title.strip(),
                "text": body.strip(),
            })
    return chunks


def embed_texts(texts):
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])


def search(question, chunks, chunk_vectors, top_k=3):
    q = embed_texts([question])[0]
    scores = chunk_vectors @ q / (
        np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    )
    best = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best]


def answer(question, chunks, vectors):
    hits = [
        (c, s) for c, s in search(question, chunks, vectors)
        if s >= MIN_SCORE
    ]
    if not hits:
        return "I don't know based on the documents I have.", [], 200

    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Keep the answer short.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL, contents=prompt
            )
            return response.text, [c["title"] for c, _ in hits], 200
        except errors.ClientError:
            # Daily quota or rate limit hit. Retrying won't help, so fail fast.
            return "Daily quota reached for the AI model. Please try again tomorrow.", [], 429
        except errors.ServerError:
            if attempt == 3:
                return "The AI model is busy right now. Try again in a minute.", [], 503
            time.sleep(2 * attempt)


if __name__ == "__main__":
    chunks = load_chunks()
    vectors = embed_texts([c["title"] + "\n" + c["text"] for c in chunks])
    print("Embedded", len(chunks), "chunks")

    while True:
        question = input("\nAsk a question (or type quit): ")
        if question.strip().lower() == "quit":
            break
        text, sources, status = answer(question, chunks, vectors)
        print(f"\n[{status}] {text}")
        if sources:
            print("Sources:", ", ".join(sources))