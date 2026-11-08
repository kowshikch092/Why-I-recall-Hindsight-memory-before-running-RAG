from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.memory.hindsight_memory import (
    recall_memory,
    store_memory
)
from src.rag.retriever import retrieve_documents
from src.tools.tavily_search import search_web
from src.llm import extract_customer_memory


groq_client = Groq(api_key=GROQ_API_KEY)


def retrieve_memory_node(state):
    try:
        memories = recall_memory(
            state["customer_id"],
            state["question"]
        )
    except Exception:
        # Hindsight being unreachable should degrade the answer quality,
        # not crash the whole support request.
        memories = []

    return {"memories": memories}


def retrieve_rag_node(state):
    try:
        documents = retrieve_documents(
            state["question"],
            top_k=4
        )
    except Exception:
        # A local Chroma/embedding failure should fall back to web
        # search rather than crash the whole support request.
        documents = []

    # Chroma distance is a similarity-distance measure: smaller is closer.
    # Empirically measured against this project's actual knowledge-base PDF:
    # the best-matching chunk for a directly relevant query (e.g. "my wifi
    # is not connecting" against the Wi-Fi troubleshooting manual) lands
    # around 0.94-1.18, while an unrelated query lands above 1.45. 1.0 was
    # rejecting genuinely relevant chunks, so the threshold is raised to 1.3.
    relevant_documents = [
        doc for doc in documents
        if doc["distance"] < 1.3
    ]

    return {
        "documents": relevant_documents,
        "use_web": len(relevant_documents) == 0
    }


def route_after_rag(state):
    if state.get("use_web", False):
        return "web_search"

    return "generate_answer"


def web_search_node(state):
    try:
        results = search_web(state["question"])
    except Exception:
        # Tavily being unavailable should not block answer generation.
        results = []

    return {"web_results": results}


def generate_answer_node(state):
    memory_context = "\n".join(
        state.get("memories", [])
    ) or "No previous customer memories found."

    pdf_context = "\n\n".join(
        [
            (
                f"Source: {doc['source']}, "
                f"Page: {doc['page']}\n{doc['content']}"
            )
            for doc in state.get("documents", [])
        ]
    ) or "No relevant PDF information found."

    web_context = "\n\n".join(
        [
            (
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Content: {item['content']}"
            )
            for item in state.get("web_results", [])
        ]
    ) or "No web search results."

    prompt = f"""
You are SmartCare AI, a helpful and professional customer support agent.

CUSTOMER QUESTION:
{state['question']}

CUSTOMER'S PREVIOUS MEMORIES:
{memory_context}

INTERNAL PDF KNOWLEDGE:
{pdf_context}

EXTERNAL WEB SEARCH RESULTS:
{web_context}

INSTRUCTIONS:
1. Answer the customer's question clearly and politely.
2. Use relevant customer memories to personalize the answer.
3. Prefer internal PDF knowledge for company-specific procedures.
4. Use web results only when relevant. Do not treat web results as
   official company policy.
5. If information is insufficient, say so. Never invent facts.
6. Give troubleshooting steps in a numbered list when useful.
7. Do not claim a troubleshooting solution worked unless the
   customer has confirmed that it worked.
8. If you use PDF information, mention its source and page where possible.
9. If external information is used, include the relevant source URL.

Write a useful answer for the customer.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SmartCare AI, an accurate customer "
                    "support assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return {"answer": answer}


def update_memory_node(state):
    """
    Distill this turn into durable customer-support facts (issue,
    environment, attempted steps, confirmed vs. unconfirmed outcomes)
    instead of storing the raw question/answer transcript, so the memory
    bank accumulates reusable facts rather than full chat logs. Falls back
    to skipping the write if extraction fails or finds nothing worth
    keeping, rather than polluting memory with low-signal content.
    """

    question = state.get("question", "").strip()
    answer = state.get("answer", "").strip()

    if not question or not answer:
        return {"new_memory": ""}

    conversation = (
        f"Customer: {question}\n"
        f"SmartCare: {answer}\n"
        "(Note: the customer has not confirmed whether any suggested "
        "troubleshooting step actually worked.)"
    )

    try:
        extracted = extract_customer_memory(conversation).strip()
    except Exception:
        extracted = ""

    if not extracted or extracted.upper().startswith("NONE"):
        return {"new_memory": ""}

    memory_text = f"Customer {state['customer_id']}:\n{extracted}"

    try:
        store_memory(
            state["customer_id"],
            memory_text
        )
    except Exception:
        # The answer was already generated and shown to the customer;
        # failing to persist the memory should not surface as an app error.
        pass

    return {"new_memory": memory_text}