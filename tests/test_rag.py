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


def test_load_all_chunks_combines_all_sources():
    expected = (
        len(rag.load_doc_chunks())
        + len(rag.load_log_chunks())
        + len(rag.load_code_chunks())
    )
    assert len(rag.load_all_chunks()) == expected


def test_load_all_chunks_includes_every_source_type():
    types = {c["type"] for c in rag.load_all_chunks()}
    assert types == {"doc", "log", "code"}


def test_load_code_chunks_splits_by_function(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os\n\nX = 1\n\n\n"
        "def foo():\n    return 1\n\n\n"
        "@app.get('/x')\ndef bar():\n    return 2\n",
        encoding="utf-8",
    )
    chunks = rag.load_code_chunks(files=(str(sample),))
    titles = [c["title"] for c in chunks]

    assert len(chunks) == 3  # setup block + foo + bar
    assert all(c["type"] == "code" for c in chunks)
    assert "def foo()" in titles
    assert "@app.get('/x') def bar()" in titles


def test_find_latest_error_returns_a_line_with_error():
    latest = rag.find_latest_error()
    assert latest is not None
    assert "ERROR" in latest