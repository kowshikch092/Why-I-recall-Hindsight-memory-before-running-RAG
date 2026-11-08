from sentence_transformers import SentenceTransformer

from src.rag.ingest import collection

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_documents(question, top_k=4):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = []

    found_docs = results.get("documents", [[]])[0]
    found_metadata = results.get("metadatas", [[]])[0]
    found_distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(
        found_docs,
        found_metadata,
        found_distances
    ):
        documents.append({
            "content": text,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": float(distance)
        })

    return documents