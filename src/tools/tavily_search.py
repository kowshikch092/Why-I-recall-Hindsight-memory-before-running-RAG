from tavily import TavilyClient
from src.config import TAVILY_API_KEY


def search_web(question):
    if not TAVILY_API_KEY:
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)

    response = client.search(
        query=question,
        search_depth="basic",
        max_results=4
    )

    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0)
        })

    return results