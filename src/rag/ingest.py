import fitz
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR

COLLECTION_NAME = "customer_support_knowledge"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def split_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_pdf(pdf_path):
    pdf_path = str(pdf_path)
    document = fitz.open(pdf_path)

    ids = []
    texts = []
    metadatas = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text")

        for chunk_number, chunk in enumerate(
            split_text(page_text)
        ):
            ids.append(
                f"{pdf_path}-{page_number}-{chunk_number}"
            )
            texts.append(chunk)
            metadatas.append({
                "source": pdf_path,
                "page": page_number
            })

    if not texts:
        return 0

    # Remove older chunks from this same PDF before re-indexing.
    try:
        collection.delete(where={"source": pdf_path})
    except Exception:
        pass

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(texts)