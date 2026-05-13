"""
Build the vector index from the Upwork API documentation PDF.
Run this once before starting the Streamlit app.

Usage:
    python build_index.py <path_to_pdf>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingest import sanity_check, chunk_documents
from rag.embeddings import create_vector_store


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <path_to_pdf>")
        print("Example: python build_index.py './docs/API_Documentation_Partial.pdf'")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store")

    print("=" * 60)
    print("STEP 1: Data Ingestion & Sanity Check")
    print("=" * 60)
    pages = sanity_check(pdf_path)

    print("\n" + "=" * 60)
    print("STEP 2: Document Chunking (500 chars, 50 overlap)")
    print("=" * 60)
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks")
    print(f"Sample chunk [0]: {chunks[0]['text'][:200]}...")
    print(f"Metadata: {chunks[0]['metadata']}")

    print("\n" + "=" * 60)
    print("STEP 3: Vector Storage (FAISS + all-MiniLM-L6-v2)")
    print("=" * 60)
    index_data = create_vector_store(chunks, persist_dir)
    print(f"Total vectors stored: {len(index_data['texts'])}")

    print("\n" + "=" * 60)
    print("INDEX BUILT SUCCESSFULLY")
    print(f"Stored at: {persist_dir}")
    print("You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
