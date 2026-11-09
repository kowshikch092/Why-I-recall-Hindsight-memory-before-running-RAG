# Why I recall Hindsight memory before running RAG

The message that changed how I ordered my support agent's pipeline was four words long: "hii". It came from a customer who, two days earlier, had spent a whole conversation explaining that his Dell laptop on Windows 11 kept losing Wi-Fi during online classes, and the agent's reply only made sense because it looked up *him* before it looked up *the manual*.

This post is about that ordering decision in SmartCare AI, a customer support agent I built for Wi-Fi troubleshooting. Each turn has two retrieval steps: one over a company knowledge base, one over long-term memory of the customer. Which one runs first and what each is allowed to contain turned out to matter more than which embedding model I picked.

## What the system does

SmartCare AI answers customer support questions about Wi-Fi problems on Windows laptops. Each turn runs through a small [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

1. **Recall** the customer's long-term memory from [Hindsight](https://github.com/vectorize-io/hindsight).
2. **Retrieve** relevant sections of the troubleshooting manual from ChromaDB.
3. **Fall back** to web search (Tavily) only if the manual has nothing close enough.
4. **Generate** an answer with an LLM on Groq, using memory, manual, and web context.
5. **Retain** a distilled set of facts from this turn back into Hindsight.

The UI is Streamlit. Full chat transcripts live in SQLite, keyed by thread. Customer memory lives in Hindsight, keyed by customer. A customer can open as many threads as they want, and all of them share one memory bank.

The wiring is deliberately boring:

```python
builder.add_edge(START, "retrieve_memory")
builder.add_edge("retrieve_memory", "retrieve_rag")

builder.add_conditional_edges(
    "retrieve_rag",
    route_after_rag,
    {
        "web_search": "web_search",
        "generate_answer": "generate_answer"
    }
)

builder.add_edge("web_search", "generate_answer")
builder.add_edge("generate_answer", "update_memory")
builder.add_edge("update_memory", END)
```

`retrieve_memory` comes first and `update_memory` comes last. Everything in this post is about why those two nodes sit at the edges of the graph and not somewhere in the middle.

## Two kinds of knowledge that don't belong together

Most RAG tutorials treat retrieval as one problem: embed the question, find similar chunks, stuff them into the prompt. A support agent has two very different kinds of knowledge to retrieve:

- **General knowledge.** "If only one laptop can't connect, check its adapter and driver." This is true for every customer, rarely changes, and comes from a document someone wrote on purpose.
- **Customer knowledge.** "This person has a Dell Inspiron on Windows 11. They already restarted the router and it didn't help. The problem started after a Windows update. They want short instructions." This is true for exactly one person, changes every conversation, and nobody wrote it down. It came out of a chat.

Early on I was tempted to push both into the same vector store with a `customer_id` metadata filter. I'm glad I didn't. The two kinds of knowledge have opposite lifecycles. The manual is re-indexed when the document changes. Customer facts are appended after every turn, need to separate "confirmed" from "suggested", and need to be scoped and deletable per person. That is a memory problem, not a document-search problem, and it's exactly what [Hindsight's agent memory API](https://hindsight.vectorize.io/) is built around: banks, `retain`, and `recall`.

So each customer gets their own bank:

```python
def get_bank_id(customer_id):
    # Keep bank IDs predictable and separate per customer.
    safe_id = "".join(
        c if c.isalnum() or c in "-_" else "-"
        for c in customer_id.lower()
    )
    return f"smartcare-{safe_id[:50]}"
```

Recall can only ever see one customer's bank, so isolation is structural rather than a filter I might forget to apply. The bank is created with a background description that tells Hindsight what this memory is for: "customer environment, recurring issues, preferences, and confirmed troubleshooting outcomes."

## The core problem: returning customers ask vague questions

Here is the part I didn't expect. First-contact questions are great RAG queries. A new customer writes a paragraph: device, OS, symptom, when it started. That embeds nicely and lands right on the "Wi-Fi disconnects repeatedly" section of the manual.

Returning customers don't do that. They write "hii", or "it's happening again", or "any next step for my issue?" Those are terrible retrieval queries. When I measured them against the manual index, "Hi, I am back. Any next step for my issue?" scored a Chroma distance of about 1.53 against its best chunk. My relevance cutoff is 1.3, so the agent decided the manual had nothing to offer and went to the web for a question that was entirely about a documented Wi-Fi problem.

The threshold isn't the bug; it is doing its job:

```python
# Empirically measured against this project's actual knowledge-base PDF:
# the best-matching chunk for a directly relevant query lands
# around 0.94-1.18, while an unrelated query lands above 1.45.
relevant_documents = [
    doc for doc in documents
    if doc["distance"] < 1.3
]

return {
    "documents": relevant_documents,
    "use_web": len(relevant_documents) == 0
}
```

The bug is asking a document index to resolve "my issue". Only memory knows what "my issue" is. That is the whole argument for recalling first:

- **Memory gives the question its meaning.** Recall returns "repeated disconnects on a Dell Inspiron, Windows 11, started after a Windows update". The retrieval step can then look up the manual for *that* instead of for "hii". In the finished system the retrieval query is the customer's message combined with the top recalled facts, so the manual lookup sees the real problem even when the customer doesn't restate it.
- **Memory decides which parts of the manual still apply.** The manual's first step for repeated disconnects is to restart the router. If memory says the customer already did that and it didn't help, the right answer starts at step four: check Windows Update and the adapter driver. RAG can't know that. It returns the same chunk to every customer.
- **Memory carries preferences the manual can't.** "Keep the steps short, I'm not technical" changes the shape of every answer after the first one.

With recall first, the generation prompt receives the customer's history and the manual side by side. The instruction is simple: use the memories to personalize, prefer the manual for procedures, and never invent facts.

The "hii" example from the opening is real. In a brand-new thread, with nothing but "hii" in the message, the agent greeted the customer by name and asked whether he wanted to continue working on the Wi-Fi drops affecting his Dell laptop during online classes. It offered to keep the steps simple, because that was a preference he'd stated in an earlier session. None of that was in the thread. All of it came from Hindsight.

## What goes into memory matters more than retrieval

The other half of the ordering is the last node. My first version of `update_memory` retained the raw question and answer. That worked for a day, and then the memory bank was mostly my agent's own long troubleshooting essays. Recall started returning the agent's suggestions back to itself.

The current version runs an extraction pass before retaining:

```python
conversation = (
    f"Customer: {question}\n"
    f"SmartCare: {answer}\n"
    "(Note: the customer has not confirmed whether any suggested "
    "troubleshooting step actually worked.)"
)

extracted = extract_customer_memory(conversation).strip()

if not extracted or extracted.upper().startswith("NONE"):
    return {"new_memory": ""}

store_memory(state["customer_id"], f"Customer {state['customer_id']}:\n{extracted}")
```

The extraction prompt asks for issues, attempted steps, failed solutions, solutions *confirmed by the customer*, preferences, and unresolved problems. It tells the model to skip greetings and generic advice. If nothing is worth keeping, it returns `NONE` and nothing is written. That's why "hii" turns don't pollute the bank.

The painful lesson here was the difference between *suggested* and *attempted*. Before I made that distinction explicit, I watched the agent open a reply with "Since we've already restarted the router, disabled power saving on the adapter, run the network troubleshooter and flushed the DNS cache…" The customer had done none of those things. The agent had *suggested* them in a previous session, the memory recorded them as steps, and the next turn read them as history. The output was fluent, confident, and wrong.

The fix is partly in extraction, which labels outcomes as confirmed or not yet confirmed, and partly in generation. A memory that says a step was suggested is not evidence that it was done. Hindsight stores what you give it. If you give it your agent's guesses, it will hand them back, perfectly recalled. Garbage in, garbage recalled.

## Keeping the manual out of memory and memory out of the manual

One more failure taught me the boundary runs both ways. The troubleshooting manual had a section, written as guidance for the agent, with *example* memory entries like "Customer C101 … router restart was attempted and did not resolve the issue" and "the customer confirmed that updating the wireless adapter driver resolved the issue."

That section got indexed into the RAG store along with everything else, and it matched a lot of Wi-Fi queries. So the LLM was being handed made-up customer history through the *document* channel. For a brand-new customer with an empty memory bank, it could still say things that sounded like "last time, the driver update worked."

The fix was to strip that section at ingest time. The broader rule I took from it: anything phrased as a fact about a person belongs in Hindsight, scoped to that person, or nowhere. The manual should only contain things that are true for everyone.

## Small engineering note: async client, sync UI

The Hindsight Python client is async, and Streamlit reruns your script top to bottom on every interaction. Creating a fresh event loop per call, or calling `asyncio.run` inside Streamlit, gave me closed-loop errors from an HTTP session that was bound to a loop that no longer existed. The solution was a single long-lived event loop on a daemon thread, with the client created lazily on that loop and every operation submitted with `run_coroutine_threadsafe`. It's not glamorous, but recall and retain now behave like plain function calls from the graph nodes. They also fail into a degraded answer instead of a stack trace: if Hindsight is unreachable, `retrieve_memory_node` returns an empty list and the agent answers from the manual alone.

## Results

Here's how the system behaves:

- **First contact:** the customer describes the problem in full. Memory recall returns nothing, the manual section for their symptom is retrieved and cited by page, and the answer is correct but generic. Retain stores device, OS, symptom, trigger, and preferences.
- **Outcome report:** "I restarted the router and laptop, it still drops." The agent moves on to the update and driver checks, and the failed restart is retained as a confirmed outcome.
- **Return visit, new thread:** "My Wi-Fi is acting up again." Recall supplies the device, the history, and the failed restart. The answer skips the restart and goes straight to the driver checks, in short steps.
- **Same message, different customer ID:** an empty bank, so the agent starts from diagnostic questions. Comparing the two answers is the clearest way I've found to show what memory adds. Same model, same manual, same prompt; only the bank differs.
- **After a confirmed fix:** "The driver update through Windows Update fixed it." Weeks later, "what worked for me last time?" gets a direct answer.

None of that requires a smarter model. It requires the model to see the right facts about the right person before it decides what the question means.

## Lessons learned

1. **Recall before you retrieve.** Returning users write underspecified messages. Memory is what turns "my issue" into a query your document index can answer, and it tells you which parts of the retrieved document no longer apply.
2. **Keep general knowledge and personal knowledge in separate systems.** Documents go in your RAG index. Facts about people go in per-user memory banks. When they bleed into each other in either direction, the model can't tell a policy from a person's history.
3. **Retain facts, not transcripts.** An extraction step that can return `NONE` keeps the bank dense and keeps recall from echoing the agent's own prose. Transcripts already have a home: SQLite, per thread.
4. **Model confirmation explicitly.** "Suggested", "attempted", and "confirmed fixed" are three different facts. If your memory doesn't distinguish them, your agent will eventually tell a customer they already did something they never did.
5. **Make memory isolation structural.** One bank per customer, derived deterministically from the customer ID, is simpler and safer than one shared index with filters.

If you're building anything where users come back, it's worth reading up on [what agent memory is and how it differs from RAG](https://vectorize.io/what-is-agent-memory) before you reach for a second vector collection. For me, separating "what's true for everyone" from "what's true for this person" and querying the second one first was the most useful design decision in the project.
