# SmartCare AI

A customer support chatbot with **persistent, per-customer long-term memory**, built for
**HackwithHyderabad 3.0 — "AI Agents That Learn Using Hindsight."**

Most support bots are stateless: a customer has to re-explain their device, their history,
and what they already tried, every single session. SmartCare AI fixes that by giving every
customer a durable memory bank (via [Hindsight](https://hindsight.vectorize.io/)) that the
agent recalls from and writes to on every turn, on top of a company knowledge-base lookup
(RAG) and a web-search fallback.

## How Hindsight memory is used

This is the mandatory, judged part of the project, so it's documented explicitly:

- **One memory bank per customer.** `src/memory/hindsight_memory.py:get_bank_id()` turns a
  customer/thread ID into a stable, sanitized Hindsight `bank_id`
  (`smartcare-<sanitized-id>`). `initialize_customer()` creates that bank on first contact
  via `client.acreate_bank(...)`, with a background description telling Hindsight what kind
  of facts to retain (recurring issues, preferences, confirmed troubleshooting outcomes).
- **Recall, before answering.** On every incoming question, `retrieve_memory_node`
  (`src/agent/nodes.py`) calls `recall_memory()`, which runs `client.arecall(bank_id, query,
  budget="low")` — a semantic search over that customer's past interactions — and returns
  the matched memory text. This is injected into the LLM prompt as
  `CUSTOMER'S PREVIOUS MEMORIES`, so the answer can reference what the agent already knows
  about that customer.
- **Distilled write-back, after answering.** `update_memory_node` no longer stores the raw
  question/answer transcript. It runs the conversation through
  `extract_customer_memory()` (`src/llm.py`) — an LLM extraction step that pulls out only
  the customer-specific facts worth keeping (issue, environment, what was tried, what was
  confirmed vs. unconfirmed) and discards greetings/generic chatter. The result is written
  with `client.aretain(bank_id, content=..., context=...)`. If extraction finds nothing
  useful, nothing is written — this keeps the memory bank signal-dense instead of bloating
  it with full chat logs.
- **No false confirmations.** Both the generation prompt and the memory-extraction prompt
  are explicitly instructed never to record (or imply) that a suggested fix worked unless
  the customer said so. This prevents the agent's own memory from later treating an
  unconfirmed guess as an established fact.

Net effect: ask SmartCare AI the same kind of question across two sessions for the same
customer ID, and the second answer is shaped by what it already knows — not a cold start.

## Architecture

```
User question (Streamlit)
        |
        v
  retrieve_memory   -> Hindsight: recall this customer's past facts
        |
        v
  retrieve_rag       -> ChromaDB: search the PDF knowledge base
        |
   (nothing relevant?)
        |--- yes --> web_search -> Tavily: live web fallback
        |--- no  ------------------------+
        v                                v
                generate_answer  -> Groq LLM (memory + PDF + web context)
                        |
                        v
                update_memory   -> Hindsight: distill & store new facts
```

Implemented as an explicit LangGraph state machine: `src/agent/graph.py` (graph wiring),
`src/agent/nodes.py` (node logic), `src/agent/state.py` (shared state schema).

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph (`StateGraph`) |
| LLM | Groq (`openai/gpt-oss-120b` by default) |
| Long-term memory | Hindsight (`hindsight-client`) |
| Knowledge-base retrieval | ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| PDF parsing | PyMuPDF |
| Web search fallback | Tavily |
| Chat history | SQLite |

## Setup

1. `python -m venv venv` and activate it.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your own `GROQ_API_KEY`, `HINDSIGHT_API_KEY`,
   and `TAVILY_API_KEY`. **Never commit `.env`** — it's git-ignored.
4. Index the knowledge-base PDF(s) in `data/` into Chroma (required once, and again any time
   you add/replace a PDF):
   ```
   python -m src.ingest_knowledge
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```

## Known limitations

- The RAG relevance threshold (`distance < 1.3` in `retrieve_rag_node`,
  `src/agent/nodes.py`) was tuned empirically against the one PDF shipped in `data/`. If you
  swap in a different knowledge base, re-measure distances for a few known-relevant queries
  and adjust the threshold.
- Hindsight recall has been observed to occasionally miss a fact written moments earlier
  (likely indexing latency) — re-asking after a short delay recalls it correctly. Don't
  assume every single recall call will surface everything ever stored.
- `customer_id` is a free-text field with no authentication — anyone can type another
  customer's ID and see/continue their conversation. Fine for a hackathon demo with
  synthetic data; not suitable as-is for production without real auth.
- No automated test suite yet.

## Demo script

1. Start a fresh "Thread ID" (e.g. `rajesh-demo`) and describe a Wi-Fi issue with some
   specific detail (laptop model, OS). Watch the PDF-knowledge panel show the matched
   manual section.
2. Ask a follow-up in a **new** browser session (or after reloading) using the same Thread
   ID, referencing "the issue I mentioned earlier" — the `🧠 Customer memories` panel shows
   what was recalled, and the answer should reference it without the customer repeating
   themselves.
3. Ask something outside the PDF's scope — watch it fall back to the `🌐 Web search
   results` panel instead of guessing.
