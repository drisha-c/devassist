from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "data" / "docs"
LOGS_DIR = BASE_DIR / "data" / "logs"
CODE_FILES = (BASE_DIR / "main.py", BASE_DIR / "rag.py")