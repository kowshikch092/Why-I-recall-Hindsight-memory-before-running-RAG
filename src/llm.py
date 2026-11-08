from groq import Groq
from src.config import GROQ_API_KEY,GROQ_MODEL
client=Groq(api_key=GROQ_API_KEY)
def generate_response(prompt):
    response=client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
             "role":"user",
             "content":prompt   
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
def generate_chat_title(question):
    fallback = " ".join(question.strip().split()[:6]) or "New Conversation"

    try:
        prompt = (
            "Summarize this customer support question as a short chat "
            "title: 3-6 words, title case, no quotes, no trailing "
            "punctuation. Return only the title, nothing else.\n\n"
            f"Question: {question.strip()}"
        )
        title = generate_response(prompt).strip().strip('"').strip()

        if not title or len(title) > 60:
            return fallback

        return title
    except Exception:
        # A flaky LLM call should never block chat creation.
        return fallback


def extract_customer_memory(conversation):
    prompt=f"""
You are a customer-support memory extraction system.
Extract only useful, customer-specific facts from this conversation.
Look for:
- Customer issues
- Previous complaints
- Troubleshooting steps attempted
- Failed solutions
- Successful solutions confirmed by the customer
- Customer preferences
- Unresolved problems
Do not save:
- Greetings
- Generic AI answers
- Repeated information without new value
- Unsupported assumptions
If no meaningful information exists, return NONE.
Conversation:
{conversation}

Return concise memory facts as bullet points.
Clearly distinguish confirmed outcomes from unconfirmed outcomes.

"""
    return generate_response(prompt)