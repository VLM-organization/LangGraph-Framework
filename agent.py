from langgraph.graph import StateGraph, END
from typing import Dict, Any
from utils import generate_response_with_llm, generate_email_with_llm
import smtplib
from email.mime.text import MIMEText
from config import EMAIL_USER, EMAIL_PASS
import time
import logging

logging.basicConfig(filename="langgraph_performance.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def agent_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response and check if an email should be sent."""
    query = state["query"]
    
    start_time = time.time()
    response = generate_response_with_llm(query)
    end_time = time.time()
    
    state["response"] = response
    state["response_time"] = end_time - start_time

    logging.info(f"Agent Step - Query: {query} | Response Time: {state['response_time']:.4f} sec")

    # Detect if email should be sent
    if "send an email" in query.lower() or "email" in query.lower():
        state["send_email"] = True

    return state


def email_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and send an email if required."""
    if state.get("send_email") and state.get("email"):
        recipient = state["email"]
        subject = "Customer Support Response"
        
        start_time = time.time()
        body = generate_email_with_llm(state["response"])
        email_gen_time = time.time() - start_time

        state["email_gen_time"] = email_gen_time
        logging.info(f"Email Generation Time: {email_gen_time:.4f} sec")

        msg = MIMEText(body)
        msg["From"] = EMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = subject

        try:
            send_start = time.time()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_USER, EMAIL_PASS)
                server.sendmail(EMAIL_USER, recipient, msg.as_string())
            send_time = time.time() - send_start

            state["email_status"] = {"status": "success", "message": "Email sent successfully."}
            state["email_send_time"] = send_time
            logging.info(f"Email Sent - Recipient: {recipient} | Send Time: {send_time:.4f} sec")

        except Exception as e:
            state["email_status"] = {"status": "error", "message": str(e)}
            logging.error(f"Email Send Error: {e}")
    
    return state


def build_agent():
    """Construct the LangGraph agent correctly."""
    graph = StateGraph(Dict[str, Any])

    # Normal chatbot interaction
    graph.add_node("agent", agent_step)

    # Only call email step if send_email is True
    graph.add_conditional_edges(
        "agent",
        lambda state: "email_sender" if state.get("send_email") else END
    )

    graph.add_node("email_sender", email_step)
    graph.add_edge("email_sender", END)

    graph.set_entry_point("agent")
    return graph.compile()
