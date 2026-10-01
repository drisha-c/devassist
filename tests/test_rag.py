import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag


def test_load_doc_chunks_returns_list():
    chunks = rag.load_doc_chunks()
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_doc_chunks_have_required_fields():
    chunks = rag.load_doc_chunks()
    for c in chunks:
        assert "source" in c
        assert "type" in c
        assert "title" in c
        assert "text" in c
        assert c["type"] == "doc"


def test_load_log_chunks_returns_list():
    chunks = rag.load_log_chunks()
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_log_chunks_tagged_correctly():
    chunks = rag.load_log_chunks()
    for c in chunks:
        assert c["type"] == "log"


def test_load_all_chunks_combines_both():
    all_chunks = rag.load_all_chunks()
    doc_chunks = rag.load_doc_chunks()
    log_chunks = rag.load_log_chunks()
    assert len(all_chunks) == len(doc_chunks) + len(log_chunks)


def test_find_latest_error_returns_a_line_with_error():
    latest = rag.find_latest_error()
    assert latest is not None
    assert "ERROR" in latest