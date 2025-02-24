from agent import build_agent
from chroma_db import extract_text_from_pdf, chunk_text, add_chunks_to_chromadb
from config import PDF_PATH

def create_chat_interface():
    """CLI for interacting with the agent."""
    print("Welcome to the AI Support Assistant! Type 'quit' to exit.")

    agent = build_agent()

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == "quit":
            print("Goodbye!")
            break

        email = None
        if "send an email" in query.lower():
            email = input("Enter recipient email: ").strip()

        # Run the agent
        state = {"query": query, "email": email}
        result = agent.invoke(state)

        print("\nAssistant:", result["response"])
        if email:
            print("\nEmail Status:", result["email_status"])
        print("\n" + "-"*50)

if __name__ == "__main__":
    pdf_path = PDF_PATH
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pdf_text)
    add_chunks_to_chromadb(chunks, "doc1")
    create_chat_interface()
