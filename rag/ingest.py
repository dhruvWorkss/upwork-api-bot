import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_PATTERNS = [
    r"^(Authentication|OAuth2|OAuth\s*2)",
    r"^(Client Credentials Grant)",
    r"^(Authorization Code Grant)",
    r"^(Implicit Grant)",
    r"^(Refresh Token Grant)",
    r"^(Service [Aa]ccounts)",
    r"^(Operations|Common Information)",
    r"^(Error [Hh]andling)",
    r"^(Subscriptions)",
    r"^(Queries|Mutations)",
    r"^(Libraries and Tools)",
    r"^(Job Post|Offers and Contracts)",
    r"^(Rate Limit)",
]


def detect_section(text: str) -> str:
    for pattern in SECTION_PATTERNS:
        if re.search(pattern, text.strip(), re.IGNORECASE):
            return re.search(pattern, text.strip(), re.IGNORECASE).group(1)
    return ""


def load_pdf(pdf_path: str) -> list[dict]:
    """Load PDF and extract text with section-aware metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    current_section = "General"

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        lines = text.split("\n")
        for line in lines:
            detected = detect_section(line)
            if detected:
                current_section = detected

        pages.append({
            "text": text,
            "page": page_num + 1,
            "section": current_section,
        })

    doc.close()
    return pages


def chunk_documents(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """Split pages into chunks with metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_data in pages:
        text = page_data["text"]
        splits = splitter.split_text(text)
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    "page": page_data["page"],
                    "section": page_data["section"],
                    "chunk_index": i,
                },
            })

    return chunks


def sanity_check(pdf_path: str):
    """Print total character count and a sample of the text."""
    pages = load_pdf(pdf_path)
    full_text = "".join(p["text"] for p in pages)
    print(f"Total characters: {len(full_text)}")
    print(f"Total pages: {len(pages)}")
    print(f"\n--- Sample (first 500 chars) ---\n{full_text[:500]}")
    print(f"\n--- Sections detected ---")
    sections = set(p["section"] for p in pages)
    for s in sorted(sections):
        print(f"  - {s}")
    return pages


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/API_Documentation_Partial.pdf"
    pages = sanity_check(path)
    chunks = chunk_documents(pages)
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Sample chunk: {chunks[0]['text'][:200]}...")
    print(f"Sample metadata: {chunks[0]['metadata']}")
