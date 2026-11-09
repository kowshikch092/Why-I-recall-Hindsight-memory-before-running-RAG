from typing import TypedDict, List, Dict, Any
class SupportState(TypedDict, total=False):
    customer_id: str
    question: str
    memories: List[str]
    documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    use_web: bool
    answer: str
    new_memory: str