import streamlit as st

from src.agent.graph import run_support_agent
from src.history import (
    initialize_database,
    create_thread,
    get_customer_threads,
    get_latest_thread_for_customer,
    get_chat_history,
    save_message,
    update_thread_title,
    clear_chat_history
)
from src.memory.hindsight_memory import initialize_customer
from src.llm import generate_chat_title


st.set_page_config(
    page_title="SmartCare AI",
    page_icon="🤖",
    layout="wide"
)

initialize_database()

# Customer ID identifies the customer (and therefore the Hindsight memory
# bank). Thread ID identifies one conversation. A customer can have many
# threads; all of them share the same memory bank.
if "customer_id" not in st.session_state:
    st.session_state.customer_id = "Customer-001"

if "thread_id" not in st.session_state:
    existing_thread = get_latest_thread_for_customer(st.session_state.customer_id)
    st.session_state.thread_id = (
        existing_thread or create_thread(st.session_state.customer_id)
    )

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ SmartCare AI")

    st.subheader("Customer Profile")

    customer_id_input = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id,
        help="Identifies the customer across all of their conversations."
    )
    new_customer_id = customer_id_input.strip() or st.session_state.customer_id

    if new_customer_id != st.session_state.customer_id:
        st.session_state.customer_id = new_customer_id
        existing_thread = get_latest_thread_for_customer(new_customer_id)
        st.session_state.thread_id = (
            existing_thread or create_thread(new_customer_id)
        )
        st.session_state.confirm_clear = False
        st.rerun()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.thread_id = create_thread(st.session_state.customer_id)
        st.rerun()

    st.divider()

    st.subheader("Chat History")

    threads = get_customer_threads(st.session_state.customer_id)

    if threads:
        for thread in threads:
            is_active = thread["thread_id"] == st.session_state.thread_id

            if st.button(
                thread["title"],
                key=f"thread_btn_{thread['thread_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    st.session_state.thread_id = thread["thread_id"]
                    st.rerun()
    else:
        st.caption("No previous conversations yet.")

    st.divider()

    with st.expander("🗑️ Clear chat history"):
        st.caption(
            "This permanently deletes every conversation for the "
            "current customer, including the one you're viewing now."
        )
        st.session_state.confirm_clear = st.checkbox(
            "I understand, delete everything for this customer",
            value=st.session_state.confirm_clear
        )

        if st.button(
            "Clear all chat history",
            disabled=not st.session_state.confirm_clear
        ):
            clear_chat_history(st.session_state.customer_id)
            st.session_state.thread_id = create_thread(st.session_state.customer_id)
            st.session_state.confirm_clear = False
            st.rerun()

    st.divider()

    st.caption(
        "Powered by LangGraph, Groq, Hindsight, "
        "ChromaDB, Tavily and SQLite."
    )


# ---------------- MAIN PAGE ----------------

st.title("🤖 SmartCare AI")
st.write(
    "Your intelligent customer support assistant "
    "with long-term memory."
)

active_customer = st.session_state.customer_id
active_thread = st.session_state.thread_id

active_title = next(
    (t["title"] for t in get_customer_threads(active_customer)
     if t["thread_id"] == active_thread),
    "New Conversation"
)
st.caption(f"💬 {active_title}  ·  Thread: `{active_thread}`")

try:
    initialize_customer(active_customer)
except Exception as error:
    st.error(f"Could not initialize customer memory: {error}")
    st.stop()


# ---------------- CHAT HISTORY ----------------

history = get_chat_history(active_thread)

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Describe your issue or ask a question..."
)

if question:
    is_first_message = len(history) == 0

    save_message(
        active_thread,
        "user",
        question
    )

    if is_first_message:
        title = generate_chat_title(question)
        update_thread_title(active_thread, title)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("SmartCare is thinking..."):
            try:
                result = run_support_agent(
                    active_customer,
                    question
                )

                answer = result.get(
                    "answer",
                    "Sorry, I could not generate a response."
                )

                st.markdown(answer)

                # Display retrieved customer memories.
                with st.expander("🧠 Customer memories"):
                    memories = result.get("memories", [])

                    if memories:
                        for memory in memories:
                            st.write("•", memory)
                    else:
                        st.write("No previous memories found.")

                # Display PDF retrieval information.
                with st.expander("📚 PDF knowledge retrieved"):
                    documents = result.get("documents", [])

                    if documents:
                        for doc in documents:
                            st.write(
                                f"**Source:** {doc['source']}"
                            )
                            st.write(
                                f"**Page:** {doc['page']}"
                            )
                            st.write(doc["content"])
                            st.divider()
                    else:
                        st.write(
                            "No relevant PDF content was retrieved."
                        )

                # Display Tavily search results.
                with st.expander("🌐 Web search results"):
                    web_results = result.get("web_results", [])

                    if web_results:
                        for item in web_results:
                            st.markdown(
                                f"**{item['title']}**"
                            )
                            st.write(item["content"])
                            if item["url"]:
                                st.markdown(
                                    f"[Open source]({item['url']})"
                                )
                            st.divider()
                    else:
                        st.write(
                            "Web search was not needed or returned no results."
                        )

                save_message(
                    active_thread,
                    "assistant",
                    answer
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
