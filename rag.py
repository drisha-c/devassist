import os
import re
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


def load_doc_chunks(folder="docs"):
    """Chunk markdown docs by heading. Returns chunks tagged as 'doc'."""
    chunks = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("\\#", "#")

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
                "type": "doc",
                "title": title.strip(),
                "text": body.strip(),
            })
    return chunks


def load_log_chunks(folder="logs", lines_per_chunk=4):
    """Chunk log files by fixed-size line windows. Returns chunks tagged as 'log'."""
    chunks = []
    for path in Path(folder).glob("*.log"):
        lines = path.read_text(encoding="utf-8").strip().splitlines()

        for i in range(0, len(lines), lines_per_chunk):
            window = lines[i:i + lines_per_chunk]
            if not window:
                continue
            chunks.append({
                "source": path.name,
                "type": "log",
                "title": f"Log lines {i + 1}-{i + len(window)}",
                "text": "\n".join(window),
            })
    return chunks


def load_code_chunks(files=("main.py", "rag.py")):
    """Chunk Python files by top-level def/class, keeping decorators attached.
    The setup code above the first function becomes its own chunk."""
    chunks = []

    for filename in files:
        path = Path(filename)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()

        # find each top-level def/class, and pull in decorators directly above it
        starts = []  # list of (chunk_start_line, def_line)
        for i, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("class "):
                start = i
                while start > 0 and lines[start - 1].startswith("@"):
                    start -= 1
                starts.append((start, i))

        if not starts:
            continue

        # setup code before the first function (imports, constants, client)
        preamble = "\n".join(lines[:starts[0][0]]).strip()
        if preamble:
            chunks.append({
                "source": filename,
                "type": "code",
                "title": f"{filename} setup and configuration",
                "text": preamble,
            })

        for n, (start, def_line) in enumerate(starts):
            end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if not block:
                continue

            title = lines[def_line].strip().rstrip(":")
            # if a FastAPI route decorator is attached, put the route in the title
            for dec in lines[start:def_line]:
                if dec.startswith("@app."):
                    title = f"{dec.strip()} {title}"
                    break

            chunks.append({
                "source": filename,
                "type": "code",
                "title": title,
                "text": block,
            })
    return chunks


def load_all_chunks():
    """Combine chunks from all sources into one list."""
    return load_doc_chunks() + load_log_chunks() + load_code_chunks()


def embed_texts(texts):
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])


def search(question, chunks, chunk_vectors, top_k=4):
    q = embed_texts([question])[0]
    scores = chunk_vectors @ q / (
        np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    )
    best = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best]


def answer(question, chunks, vectors):
    start_time = time.time()
    all_hits = search(question, chunks, vectors)
    top_score = all_hits[0][1] if all_hits else 0.0
    hits = [(c, s) for c, s in all_hits if s >= MIN_SCORE]

    if not hits:
        elapsed_ms = round((time.time() - start_time) * 1000)
        return "I don't know based on the documents I have.", [], 200, top_score, elapsed_ms

    context = "\n\n".join(
        f"[{c['type']}: {c['title']}]\n{c['text']}" for c, _ in hits
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Give a clear, complete answer in 1-3 sentences, don't just say yes or no.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL, contents=prompt
            )
            sources = [f"{c['type']}: {c['title']}" for c, _ in hits]
            elapsed_ms = round((time.time() - start_time) * 1000)
            return response.text, sources, 200, top_score, elapsed_ms
        except errors.ClientError as e:
            elapsed_ms = round((time.time() - start_time) * 1000)
            if getattr(e, "code", None) == 429:
                return "Daily quota reached for the AI model. Please try again tomorrow.", [], 429, top_score, elapsed_ms
            return "The AI model rejected the request. Check the server's API key and model settings.", [], 502, top_score, elapsed_ms
        except errors.ServerError:
            if attempt == 3:
                elapsed_ms = round((time.time() - start_time) * 1000)
                return "The AI model is busy right now. Try again in a minute.", [], 503, top_score, elapsed_ms
            time.sleep(2 * attempt)


def find_latest_error(folder="logs"):
    """Scan log files for the most recent ERROR line."""
    error_lines = []
    for path in Path(folder).glob("*.log"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"\bERROR\b", line):
                error_lines.append(line)
    if not error_lines:
        return None
    return error_lines[-1]


def suggest_fix(chunks, vectors):
    """Find the latest error, retrieve context, and ask Gemini to draft a fix."""
    latest_error = find_latest_error()
    if not latest_error:
        return None, "No errors found in the logs.", [], 200

    query = f"How to fix this error: {latest_error}"
    hits = [
        (c, s) for c, s in search(query, chunks, vectors, top_k=4)
        if s >= MIN_SCORE
    ]

    if not hits:
        return latest_error, "No relevant context found to suggest a fix.", [], 200

    context = "\n\n".join(
        f"[{c['type']}: {c['title']}]\n{c['text']}" for c, _ in hits
    )
    prompt = (
        "You are a DevOps assistant. An error was detected in the logs. "
        "Using ONLY the context below, suggest a short, specific remediation action. "
        "If the context does not explain how to fix it, say so honestly. "
        "Keep the suggestion to 2-3 sentences.\n\n"
        f"Error: {latest_error}\n\n"
        f"Context:\n{context}"
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL, contents=prompt
            )
            sources = [f"{c['type']}: {c['title']}" for c, _ in hits]
            return latest_error, response.text, sources, 200
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                return latest_error, "Daily quota reached for the AI model.", [], 429
            return latest_error, "The AI model rejected the request. Check the server's API key and model settings.", [], 502
        except errors.ServerError:
            if attempt == 3:
                return latest_error, "The AI model is busy right now.", [], 503
            time.sleep(2 * attempt)


if __name__ == "__main__":
    chunks = load_all_chunks()
    vectors = embed_texts([c["title"] + "\n" + c["text"] for c in chunks])
    doc_count = sum(1 for c in chunks if c["type"] == "doc")
    log_count = sum(1 for c in chunks if c["type"] == "log")
    code_count = sum(1 for c in chunks if c["type"] == "code")
    print(f"Embedded {len(chunks)} chunks ({doc_count} doc, {log_count} log, {code_count} code)")

    while True:
        question = input("\nAsk a question (or type quit): ")
        if question.strip().lower() == "quit":
            break
        text, sources, status, score, ms = answer(question, chunks, vectors)
        print(f"\n[{status}] ({ms}ms, top score {score:.3f}) {text}")
        if sources:
            print("Sources:", ", ".join(sources))