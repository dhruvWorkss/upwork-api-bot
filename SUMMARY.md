# Technical Summary

## Difficulties Faced & Solutions

1. **PDF section detection**: Standard text extraction loses document structure. I used PyMuPDF with regex-based header detection to tag chunks with their source section (OAuth2, GraphQL, Subscriptions, etc.), enabling section-aware retrieval.

2. **Exact term retrieval**: Pure semantic search (embeddings) sometimes ranks paraphrased text higher than exact matches for technical terms like "Client Credentials Grant" or "access_token". I added BM25 keyword scoring fused with semantic similarity (70/30 weighted) to catch exact terminology.

3. **Hallucination prevention**: An LLM cannot reliably self-assess whether retrieved context actually answers a question. I implemented a retrieval-score threshold  - if the best chunk scores below 0.35, the system prompt forces the "not in documentation" response before the LLM even attempts to answer.

4. **API latency visibility**: DeepInfra response times vary. I added streaming with time-to-first-token tracking so users see immediate feedback, plus total latency displayed as a color-coded badge.

## How I Used LLMs in Development

I used Claude as a code architecture advisor  - discussing the tradeoff between single-file notebooks vs. modular structure, iterating on the system prompt wording to minimize hallucination, and debugging edge cases in chunk overlap behavior with code snippets.

## Why I'm the Right Fit for ProAnalyst

1. **I think about failure modes first**  - the hallucination guard, confidence scoring, and hybrid retrieval aren't features that a tutorial teaches; they come from understanding how RAG systems actually break in production.

2. **I write production-shaped code**  - modular architecture, environment variable handling, clear separation of concerns. This isn't a notebook that only works once; it's a foundation you could extend.

3. **I balance speed with quality**  - hybrid retrieval is 20 extra lines but catches cases pure embeddings miss. Streaming is slightly more complex than blocking calls but transforms the user experience. I pick high-leverage improvements, not over-engineering.
