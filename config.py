from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
LOGS_DIR = BASE_DIR / "logs"
CODE_FILES = (BASE_DIR / "main.py", BASE_DIR / "rag.py")