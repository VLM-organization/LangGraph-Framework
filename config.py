import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys & Configs
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
PDF_PATH = os.getenv("PDF_PATH")