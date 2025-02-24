from openai import OpenAI
from config import NVIDIA_API_KEY
from chroma_db import retrieve_context

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

def generate_response_with_llm(query: str) -> str:
    """Generate a response using LLM after retrieving context."""

    if not query.strip() or query.startswith("d:\\"):
        return "Hello! How can I assist you today?"

    retrieved_context = retrieve_context(query)

    llm_prompt = f"""
    You are an AI-powered customer support assistant. Your goal is to provide clear, concise, and helpful answers based on the provided knowledge base.

    ## Context:
    {retrieved_context}

    ## User Query:
    {query}

    ## Instructions:
    - Use the given context to answer the query.
    - If the context is insufficient, use general knowledge.
    - Provide structured, professional, and helpful responses.
    - If an email needs to be sent, format it formally.

    Now, generate the best possible response.
    """

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": llm_prompt}],
        temperature=0.5,
        max_tokens=1024
    )

    return completion.choices[0].message.content

def generate_email_with_llm(previous_response: str) -> str:
    """Generate a professional email based on the previous LLM response."""
    email_prompt = f"""
    You are an AI email assistant. Generate a formal email based on the provided response.

    ## Response:
    {previous_response}

    ## Instructions:
    - Keep the email professional and structured.
    - Start with a polite greeting.
    - Summarize the issue and provide the resolution.
    - End with a closing remark and signature.

    Now, generate the email.
    """

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": email_prompt}],
        temperature=0.5,
        max_tokens= 1024
    )

    return completion.choices[0].message.content
