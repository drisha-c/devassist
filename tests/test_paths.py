import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag


def test_chunks_load_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run from an empty folder
    types = {c["type"] for c in rag.load_all_chunks()}
    assert types == {"doc", "log", "code"}


def test_code_chunk_sources_are_plain_filenames():
    sources = {c["source"] for c in rag.load_code_chunks()}
    assert sources == {"main.py", "rag.py"}