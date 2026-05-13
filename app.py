import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.retriever import HybridRetriever
from rag.llm import query_llm, CONFIDENCE_THRESHOLD


st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)

st.title("Upwork API Technical Support Bot")
st.caption("RAG-powered assistant for Upwork API documentation | Senior API Consultant mode")


@st.cache_resource
def load_retriever():
    persist_dir = os.path.join(os.path.dirname(__file__), "vector_store")
    return HybridRetriever(persist_dir=persist_dir)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            col1, col2, col3 = st.columns(3)
            with col1:
                latency = meta.get("latency", 0)
                color = "green" if latency < 2 else "orange" if latency < 5 else "red"
                st.markdown(f"⏱️ :{color}[{latency:.2f}s]")
            with col2:
                confidence = meta.get("confidence", 0)
                st.markdown(f"🎯 Confidence: {confidence:.0%}")
            with col3:
                st.markdown(f"📄 Sources: {meta.get('num_sources', 0)}")

            with st.expander("📚 View Sources"):
                for source in meta.get("sources", []):
                    st.markdown(f"**[Section: {source['section']}]** (Score: {source['score']:.3f})")
                    st.code(source["text"][:300], language=None)

            with st.expander("🔍 Retrieval Debug"):
                for source in meta.get("sources", []):
                    st.markdown(
                        f"- `{source['id']}` | Semantic: {source['semantic']:.3f} | "
                        f"BM25: {source['bm25']:.3f} | Combined: {source['score']:.3f}"
                    )

# Chat input
if prompt := st.chat_input("Ask about the Upwork API..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            retriever = load_retriever()
        except Exception as e:
            st.error(f"Failed to load vector store. Run `python build_index.py` first.\n\nError: {e}")
            st.stop()

        # Retrieve relevant chunks
        retrieved = retriever.retrieve(prompt, top_k=3)
        top_score = retrieved[0]["score"] if retrieved else 0.0

        # Stream LLM response
        response_placeholder = st.empty()
        stream, start_time = query_llm(prompt, retrieved, stream=True)

        full_response = ""
        first_token_time = None

        for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")

        total_time = time.time() - start_time
        response_placeholder.markdown(full_response)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "green" if total_time < 2 else "orange" if total_time < 5 else "red"
            st.markdown(f"⏱️ :{color}[{total_time:.2f}s] (TTFT: {first_token_time:.2f}s)" if first_token_time else f"⏱️ :{color}[{total_time:.2f}s]")
        with col2:
            st.markdown(f"🎯 Confidence: {top_score:.0%}")
        with col3:
            st.markdown(f"📄 Sources: {len(retrieved)}")

        # Sources
        with st.expander("📚 View Sources"):
            for r in retrieved:
                section = r["metadata"].get("section", "Unknown")
                st.markdown(f"**[Section: {section}]** (Score: {r['score']:.3f})")
                st.code(r["text"][:300], language=None)

        with st.expander("🔍 Retrieval Debug"):
            for r in retrieved:
                st.markdown(
                    f"- `{r['id']}` | Semantic: {r['semantic_score']:.3f} | "
                    f"BM25: {r['bm25_score']:.3f} | Combined: {r['score']:.3f}"
                )
            if top_score < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Low confidence  - hallucination guard activated")

        # Store in session
        sources_meta = [
            {
                "id": r["id"],
                "text": r["text"],
                "section": r["metadata"].get("section", "Unknown"),
                "score": r["score"],
                "semantic": r["semantic_score"],
                "bm25": r["bm25_score"],
            }
            for r in retrieved
        ]

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "metadata": {
                "latency": total_time,
                "confidence": top_score,
                "num_sources": len(retrieved),
                "sources": sources_meta,
            },
        })


# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This bot answers questions about the **Upwork API** using "
        "Retrieval-Augmented Generation (RAG)."
    )
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        "- **Embeddings**: all-MiniLM-L6-v2\n"
        "- **Vector DB**: FAISS (numpy)\n"
        "- **Retrieval**: Hybrid (Semantic + BM25)\n"
        "- **LLM**: Meta-Llama-3.1-8B-Instruct-Turbo\n"
        "- **Hallucination Guard**: Score-threshold based"
    )
    st.markdown("---")
    st.markdown("### Confidence Guide")
    st.markdown(
        "- 🟢 **> 50%**: High confidence answer\n"
        "- 🟡 **35-50%**: Moderate confidence\n"
        "- 🔴 **< 35%**: Guard activated  - likely not in docs"
    )
