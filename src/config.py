import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
HINDSIGHT_BASE_URL = os.getenv(
    "HINDSIGHT_BASE_URL",
    "https://api.hindsight.vectorize.io"
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
CHROMA_DIR = DATABASE_DIR / "chroma_db"
CHAT_DB_PATH = DATABASE_DIR / "chat_history.db"

PDF_PATH = DATA_DIR / "Customer_Support_Agent_Knowledge_Base.pdf"

for folder in [DATA_DIR, DATABASE_DIR, CHROMA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in your .env file.")

if not HINDSIGHT_API_KEY:
    raise ValueError("HINDSIGHT_API_KEY is missing in your .env file.")