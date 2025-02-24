import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from typing import Dict, Any, List
from langchain_chroma import Chroma
from config import CHROMA_DB_PATH

chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]

# Initialize ChromaDB with the new embedding function
embedding_function = SentenceTransformerEmbeddings(embedding_model)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(client=chroma_client, embedding_function=embedding_function)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to chunk text
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to add text chunks to ChromaDB
def add_chunks_to_chromadb(chunks, doc_id: str):
    vectorstore.add_texts(chunks, metadatas=[{"source": doc_id} for _ in chunks])

def retrieve_context(query: str) -> str:
    """Retrieve relevant context for a given query using RAG."""
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")])