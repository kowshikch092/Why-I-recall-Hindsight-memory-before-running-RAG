from src.config import DATA_DIR
from src.rag.ingest import ingest_pdf


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data folder.")
        return

    for pdf_path in pdf_files:
        print(f"Indexing: {pdf_path.name}")

        try:
            count = ingest_pdf(pdf_path)
            print(f"Successfully indexed {count} chunks.")

        except Exception as error:
            print(f"Error indexing {pdf_path.name}: {error}")

    print("PDF indexing completed!")


if __name__ == "__main__":
    main()