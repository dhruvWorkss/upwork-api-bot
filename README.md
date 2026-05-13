# Upwork API Technical Support Bot

A RAG-powered chatbot that answers developer questions about the Upwork API using retrieval-augmented generation.

## Live Demo

[Try the app here](https://upwork-api-bot-ajz8th4adbob4cinvvhuct.streamlit.app)

## Architecture

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: NumPy + Pickle (cosine similarity search)
- **Retrieval**: Hybrid (Semantic + BM25 keyword fusion)
- **LLM**: Meta-Llama-3.1-8B-Instruct-Turbo via DeepInfra
- **UI**: Streamlit

## Features

- Hybrid retrieval combining semantic search with BM25 keyword matching
- Confidence-based hallucination guard
- Streaming responses with latency display
- Source attribution showing exact documentation snippets used
- Retrieval debug panel showing individual chunk scores

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
python build_index.py "API Documentation Partial.pdf"

# Run the app
streamlit run app.py
```

App opens at http://localhost:8501

## Project Structure

```
├── app.py              # Streamlit chat interface
├── build_index.py      # One-time PDF indexing script
├── rag/
│   ├── ingest.py       # PDF loading + section-aware chunking
│   ├── embeddings.py   # Vector store (numpy + pickle)
│   ├── retriever.py    # Hybrid retrieval (semantic + BM25)
│   └── llm.py          # DeepInfra LLM + hallucination guard
├── vector_store/       # Pre-built embeddings index
├── requirements.txt
├── .env.example
└── SUMMARY.md
```
