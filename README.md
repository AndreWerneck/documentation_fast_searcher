# Documentation Q&A Assistant

This project is a lightweight, modular proof-of-concept (PoC) system designed to assist developers in navigating large technical documentation efficiently. It targets the common pain point of developers spending too much time searching for information or relying on colleagues for answers that are often buried in internal documents.

The current implementation uses AWS documentation to simulate this scenario and demonstrates a pipeline that integrates preprocessing, hybrid document retrieval (sparse + dense), and local large language model (LLM) generation.

---

## Project Motivation

Engineering teams often face productivity bottlenecks when navigating vast and complex documentation. This project aims to:
	-	Provide instant, accurate answers to user questions based on existing documentation.
	-	Reduce interruptions to experienced engineers.
	-	Ensure responses remain consistent and up-to-date with the latest documents.

While the primary use case is search and Q&A over AWS documentation, the architecture is built with flexibility and scalability in mind, enabling easy extension to internal, private datasets.

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
```

The system follows a Retrieval-Augmented Generation (RAG) pattern, which combines a document retriever and a text generator:    
1.	Preprocessing & Chunking (preprocessing.py)
    - Markdown files are parsed into clean text, split into manageable chunks, and saved with metadata for later retrieval.
2.	Indexing Phase (build_index.py)
    -	Dense embeddings are generated using sentence-transformers.
	-	Sparse BM25 vectors are computed using rank_bm25.
	-	Metadata, chunks, and indexes are stored to disk.
3.	Retrieval Layer
	-	sparse_embedder.py: Performs keyword-based retrieval with BM25.
	-	dense_embedder.py: Retrieves semantically relevant chunks.
	-	vector_store.py: Uses FAISS to index and encapsulate logic for vector semantic search with cosine-similarity.
4.	Reranking (Late Fusion) (generator.py)
    - Retrieved results are reranked using a cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2) to select the most relevant chunks.
5.	Prompt Construction + LLM Response
    - Top-k reranked chunks are used to form the prompt.
    - Generation is handled locally via llama-cpp-python, running quantized Mistral-7B-Instruct in GGUF format from the models/ directory.

---

### Project Structure
```
.
├── data/                  # Stored embeddings, chunks, and indexes
├── models/                # Local model files (GGUF)
├── src/
│   ├── init.py                  # Source code
│   ├── preprocessing.py
│   ├── build_index.py
│   ├── vector_store.py
│   ├── dense_embedder.py
│   ├── sparse_embedder.py
│   ├── evaluate_retrieval.py
│   ├── generator.py
│   ├── main.py
│   ├── config.py
│   ├── docs_fast_searcher.ipynb  # notebook for seeing answers for suggested questions and to play with the solution
├── llm_install.py         # Local model loader using llama-cpp
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md
```

---

## How to Run the Project

### Setup

This project is made for MacOS and Linux only. 
Make sure you have Python 3.10+ installed. 
Then run:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Building the Index

Here, you have two options: If you're gonna test this repo in the dataset already present here, you can skip this step because the indexing is already made (got to Running the QA Pipeline). Otherwise, if you want to test it in another dataset or update the current one, you need to turn in command line: 

```bash
python src/build_index.py
```

This will process and chunk the raw documents, compute embeddings, and store chunks, indexes and metadata in the data/ folder. As by default I'm using semantic chunking it will take around 4 minutes for the this step (as for my machine Macbook Air M4 16gb ram 256 ssd).

### Running the QA Pipeline

You can test the full retrieval + generation flow using:

```bash
python src/main.py
```

You’ll be prompted to enter a question. After it the system will prompt the token count and the answer alongside with the main source and related files for further reading. 

--- 

## Tooling Choices Justified
-	sentence-transformers + FAISS: Efficient dense vector search; captures semantic similarity.
-	rank_bm25: Lightweight sparse (key-word) retriever to complement dense search.
-	Hybrid Retrieval + Cross-Encoder Reranker: Increases precision of retrieved results. Rerank already top-ranked chunks based on similarity with the query.
-	llama-cpp-python with Mistral-7B-Instruct GGUF: Allows fully local LLM inference, making it possible to test a good enough LLM in this POC.

This design ensures flexibility across use cases, including ones with sensitive or private data as we could easily replace some modules for more powerfull API that would ensure better speed and geographic and PPI control. 

### Design Decisions
-	Chunking Strategy: I've implemented 3 chunking strategies. Chunking by header is possible as well as chunking by tokens or characters count. Besides that semantic chunking using the same encoder used for vectorization and similarity search is also available and is the default option. 
-	Late Fusion Reranking: Dense and sparse results are merged and then re-ranked using a cross-encoder for optimal relevance wrt the query.

---

🧪 Evaluation

Evaluation scripts (evaluate_retrieval.py) allow assessment of retrieval quality using a small QA set. Metrics like hit rate, top-k coverage, and reranking gain are considered. This can be extended with ground-truth annotations or relevance feedback in a production setting.

⸻

✅ What’s Production-Ready?
	-	Modular pipeline: easy to adapt/extend.
	-	Local model loading and offline inference.
	-	Efficient document indexing and hybrid search.

🚧 What’s Missing
	-	No web interface or API (could be added via FastAPI)
	-	No continuous ingestion pipeline for updated documents.
	-	No fine-tuning or domain-specific retriever optimization.

⸻

🔄 Updating the Knowledge Base

The system can be re-indexed periodically using the same build_index.py script. Future iterations could support real-time indexing and invalidation of outdated chunks.

⸻

🧪 Tests

Basic tests are included in the tests/ folder to validate key components of the system such as the generator pipeline.

Run tests using:

pytest tests/




⸻

🤝 Contributions

Pull requests and suggestions welcome. This project is meant to grow with evolving needs in document Q&A and retrieval systems.

⸻

📜 License

MIT License.