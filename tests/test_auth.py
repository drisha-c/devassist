import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import HTTPException

import main


def test_rejects_everything_when_server_key_not_set(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        main.check_api_key(None)
    assert exc.value.status_code == 500


def test_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        main.check_api_key(None)
    assert exc.value.status_code == 401


def test_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        main.check_api_key("wrong")
    assert exc.value.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    assert main.check_api_key("secret") is None