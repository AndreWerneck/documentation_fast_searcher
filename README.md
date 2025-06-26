# Documentation Assistant – RAG-based QA System (POC)

This project is a **Retrieval-Augmented Generation (RAG)** system built as a technical proof of concept. It helps developers query and understand AWS SageMaker documentation efficiently by combining **hybrid retrieval (dense + sparse)**, **reranking**, and a **local LLM-based generator**.

## Objective

To demonstrate how (even a lightweight) RAG system can help developers reduce time spent searching documentation by providing accurate, source-grounded answers using natural language queries.

---

## Architecture Overview

```text
             ┌──────────────────┐
             │   User Query     │
             └────────┬─────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Hybrid Retrieval (Dense + │
        │ Sparse + Deduplication)   │
        └─────────────┬─────────────┘
                      │
             ┌────────▼────────┐
             │  CrossEncoder   │  → Rerank by relevance
             └────────┬────────┘
                      │
               Top-k Relevant Chunks
                      │
             ┌────────▼────────┐
             │   Prompt + LLM  │  → Mistral-7B (local)
             └────────┬────────┘
                      │
              Final Answer + Sources
