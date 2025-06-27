# Documentation Assistant – RAG-based QA System

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

# Documentation Q&A Assistant

This project is a lightweight, modular proof-of-concept (PoC) system designed to assist developers in navigating large technical documentation efficiently. It targets the common pain point of developers spending too much time searching for information or relying on colleagues for answers that are often buried in internal documents.

The current implementation uses AWS documentation to simulate this scenario and demonstrates a pipeline that integrates preprocessing, hybrid document retrieval (sparse + dense), and local large language model (LLM) generation.

⸻

🧠 Project Motivation

Engineering teams often face productivity bottlenecks when navigating vast and complex documentation. This project aims to:
	•	Provide instant, accurate answers to user questions based on existing documentation.
	•	Reduce interruptions to experienced engineers.
	•	Ensure responses remain consistent and up-to-date with the latest documents.

While the primary use case is search and Q&A over AWS documentation, the architecture is built with flexibility and scalability in mind, enabling easy extension to internal, private datasets.

⸻

🏗️ Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) pattern, which combines a document retriever and a text generator:
	1.	Preprocessing & Chunking (preprocessing.py)
Markdown files are parsed into clean text, split into manageable chunks, and saved with metadata for later retrieval.
	2.	Indexing Phase (build_index.py)
	•	Dense embeddings are generated using sentence-transformers.
	•	Sparse BM25 vectors are computed using rank_bm25.
	•	Metadata, chunks, and indexes are stored to disk.
	3.	Retrieval Layer
	•	sparse_embedder.py: Performs keyword-based retrieval with BM25.
	•	dense_embedder.py: Retrieves semantically relevant chunks with FAISS.
	•	vector_store.py: Encapsulates logic for combining retrieval outputs.
	4.	Reranking (Late Fusion) (generator.py)
Retrieved results are reranked using a cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-en-de-v1) to select the most relevant chunks.
	5.	Prompt Construction + LLM Response
Top-k reranked chunks are used to form the prompt.
Generation is handled locally via llama-cpp-python, running Mistral-7B-Instruct in GGUF format from the models/ directory.

⸻

🚀 How to Run the Project

⚙️ Setup

Make sure you have Python 3.10+ installed. Then run:

bash setup.sh

This script will:
	•	Create a virtual environment
	•	Install dependencies
	•	Download necessary NLTK packages

Alternatively, install everything manually:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

🧱 Building the Index

python src/build_index.py

This will process the raw documents, compute embeddings, and store indexes in the data/ folder.

🧪 Running the QA Pipeline

You can test the full retrieval + generation flow using:

python src/main.py

You’ll be prompted to enter a question.

⸻

🧰 Tooling Choices Justified
	•	sentence-transformers + FAISS: Efficient dense vector search; captures semantic similarity.
	•	rank_bm25: Lightweight sparse retriever to complement dense search.
	•	Hybrid Retrieval + Cross-Encoder Reranker: Increases precision of retrieved results.
	•	llama-cpp-python with Mistral-7B-Instruct GGUF: Allows fully local LLM inference, critical for privacy and geographical constraints.

This design ensures flexibility across use cases, including ones with sensitive or private data.

⸻

🧠 Design Decisions
	•	Chunking Strategy: Documents are split by paragraphs and further chunked into 300-400 token windows. This provides contextual coverage without exceeding LLM context limits.
	•	Late Fusion Reranking: Dense and sparse results are merged and then re-ranked using a cross-encoder for optimal relevance.
	•	Local Model Inference: Chosen to avoid data leakage and support air-gapped environments.

⸻

🧪 Evaluation

Evaluation scripts (evaluate_retrieval.py) allow assessment of retrieval quality using a small QA set. Metrics like hit rate, top-k coverage, and reranking gain are considered. This can be extended with ground-truth annotations or relevance feedback in a production setting.

⸻

✅ What’s Production-Ready?
	•	Modular pipeline: easy to adapt/extend.
	•	Local model loading and offline inference.
	•	Efficient document indexing and hybrid search.

🚧 What’s Missing
	•	No web interface or API (could be added via FastAPI)
	•	No continuous ingestion pipeline for updated documents.
	•	No fine-tuning or domain-specific retriever optimization.

⸻

🔄 Updating the Knowledge Base

The system can be re-indexed periodically using the same build_index.py script. Future iterations could support real-time indexing and invalidation of outdated chunks.

⸻

🧪 Tests

Basic tests are included in the tests/ folder to validate key components of the system such as the generator pipeline.

Run tests using:

pytest tests/


⸻

📁 Project Structure

.
├── data/                  # Stored embeddings, chunks, and indexes
├── models/                # Local model files (GGUF)
├── src/                   # Source code
│   ├── preprocessing.py
│   ├── build_index.py
│   ├── vector_store.py
│   ├── dense_embedder.py
│   ├── sparse_embedder.py
│   ├── generator.py
│   ├── main.py
│   ├── config.py
├── llm_install.py         # Local model loader using llama-cpp
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md


⸻

🤝 Contributions

Pull requests and suggestions welcome. This project is meant to grow with evolving needs in document Q&A and retrieval systems.

⸻

📜 License

MIT License.