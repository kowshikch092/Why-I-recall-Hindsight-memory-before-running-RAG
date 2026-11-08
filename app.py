import streamlit as st

from src.agent.graph import run_support_agent
from src.history import (
    initialize_database,
    save_message,
    get_chat_history,
    get_all_customers,
    clear_chat_history
)
from src.memory.hindsight_memory import initialize_customer


st.set_page_config(
    page_title="SmartCare AI",
    page_icon="🤖",
    layout="wide"
)

initialize_database()

if "customer_id" not in st.session_state:
    st.session_state.customer_id = "Thread-001"

if "processing" not in st.session_state:
    st.session_state.processing = False


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ SmartCare AI")

    st.subheader("Customer Profile")

    customer_id = st.text_input(
        "Thread ID",
        value=st.session_state.customer_id
    )

    if customer_id.strip():
        st.session_state.customer_id = customer_id.strip()

    st.divider()

    st.subheader("Chat History")

    customers = get_all_customers()

    if customers:
        selected_customer = st.selectbox(
            "Select previous Chat",
            customers,
            index=(
                customers.index(st.session_state.customer_id)
                if st.session_state.customer_id in customers
                else 0
            )
        )

        if st.button("Load selected Chats"):
            st.session_state.customer_id = selected_customer
            st.rerun()

    if st.button("Clear Previous chat's  history"):
        clear_chat_history(st.session_state.customer_id)
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

current_customer = st.session_state.customer_id

try:
    initialize_customer(current_customer)
except Exception as error:
    st.error(f"Could not initialize customer memory: {error}")
    st.stop()


# ---------------- CHAT HISTORY ----------------

history = get_chat_history(current_customer)

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Describe your issue or ask a question..."
)

if question:
    save_message(
        current_customer,
        "user",
        question
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("SmartCare is thinking..."):
            try:
                result = run_support_agent(
                    current_customer,
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
                    current_customer,
                    "assistant",
                    answer
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")