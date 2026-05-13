import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with deep expertise in OAuth2, GraphQL, and the Upwork platform. You answer developer questions accurately and concisely based ONLY on the provided documentation context.

Rules:
1. Only answer based on the provided context below. Never invent information.
2. If the context does not contain the answer, respond exactly: "I'm sorry, but the provided documentation does not contain that information."
3. When citing specifics (endpoints, TTLs, grant types), quote the relevant part of the documentation.
4. Be direct and technical. Developers don't need hand-holding.
5. Structure your answer clearly with brief explanations."""

HALLUCINATION_GUARD_PROMPT = """IMPORTANT: The retrieval system found LOW confidence matches for this query. The documentation likely does NOT contain this information. You MUST respond with: "I'm sorry, but the provided documentation does not contain that information." Do NOT attempt to answer from general knowledge."""

CONFIDENCE_THRESHOLD = 0.30


def get_llm_client() -> OpenAI:
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY not set in environment variables")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )


def build_prompt(query: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Build the message list for the LLM with context from retrieved chunks."""
    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

    system = SYSTEM_PROMPT
    if top_score < CONFIDENCE_THRESHOLD:
        system += "\n\n" + HALLUCINATION_GUARD_PROMPT

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        section = chunk["metadata"].get("section", "Unknown")
        context_parts.append(f"[Source {i} | Section: {section}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Documentation Context:\n{context_block}\n\n---\n\nDeveloper Question: {query}"},
    ]
    return messages


def query_llm(query: str, retrieved_chunks: list[dict], stream: bool = True):
    """Query the LLM and return response with timing info."""
    client = get_llm_client()
    messages = build_prompt(query, retrieved_chunks)

    start_time = time.time()

    if stream:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )
        return response, start_time
    else:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            stream=False,
            temperature=0.1,
            max_tokens=1024,
        )
        elapsed = time.time() - start_time
        return response.choices[0].message.content, elapsed
