# Upwork API Technical Support Bot

A RAG-powered chatbot that answers developer questions about the Upwork API using retrieval-augmented generation.

## Architecture

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: Hybrid (Semantic + BM25 keyword fusion)
- **LLM**: Meta-Llama-3.1-8B-Instruct-Turbo via DeepInfra
- **UI**: Streamlit

## Setup & Run (Local)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env with your API key
echo DEEPINFRA_API_KEY=your_key_here > .env

# Build the vector index (run once)
python build_index.py "path/to/API Documentation Partial.pdf"

# Run the app
streamlit run app.py
```

App opens at http://localhost:8501

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app"
5. Select your repo, branch `main`, file `app.py`
6. Under "Advanced settings" → add secret: `DEEPINFRA_API_KEY = your_key_here`
7. Click "Deploy"

**Important**: Before deploying, you must build the `chroma_db/` index locally and push it to the repo (or add index-building to the app startup).

## Project Structure

```
├── app.py              # Streamlit chat interface
├── build_index.py      # One-time PDF indexing script
├── rag/
│   ├── ingest.py       # PDF loading + section-aware chunking
│   ├── embeddings.py   # ChromaDB vector store
│   ├── retriever.py    # Hybrid retrieval (semantic + BM25)
│   └── llm.py          # DeepInfra LLM + hallucination guard
├── requirements.txt
├── .env.example
└── SUMMARY.md
```
