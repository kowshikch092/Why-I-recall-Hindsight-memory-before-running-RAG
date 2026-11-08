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