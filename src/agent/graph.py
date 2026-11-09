from langgraph.graph import StateGraph, START, END

from src.agent.state import SupportState
from src.agent.nodes import (
    retrieve_memory_node,
    retrieve_rag_node,
    route_after_rag,
    web_search_node,
    generate_answer_node,
    update_memory_node
)


def build_support_graph():
    builder = StateGraph(SupportState)

    builder.add_node(
        "retrieve_memory",
        retrieve_memory_node
    )

    builder.add_node(
        "retrieve_rag",
        retrieve_rag_node
    )

    builder.add_node(
        "web_search",
        web_search_node
    )

    builder.add_node(
        "generate_answer",
        generate_answer_node
    )

    builder.add_node(
        "update_memory",
        update_memory_node
    )

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "retrieve_rag")
    '''
    retrieve_rag
   ├─ use_web is true  → web_search → generate_answer
   └─ otherwise        → generate_answer'''
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

    return builder.compile()


support_graph = build_support_graph()


def run_support_agent(customer_id, question):
    initial_state = {
        "customer_id": customer_id,
        "question": question,
        "memories": [],
        "documents": [],
        "web_results": [],
        "use_web": False,
        "answer": "",
        "new_memory": ""
    }

    return support_graph.invoke(initial_state)