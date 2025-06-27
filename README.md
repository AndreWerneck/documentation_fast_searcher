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

## How to Run the Project

First of all, clone this repo, enter the repo folder and then follow the instructions given below.  

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
### Installing the LLM

Run the command line below:

```bash
python llm_install.py
```

This may take sometime depeding on your machine. Please, be patient.

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

## Evaluation

Retrieval quality can be assessed using the **evaluate_retrieval.py** script, which returns top results from both dense and sparse retrievers for a given query. This enabled human-in-the-loop validation throughout development using a sort of hit rate inspection. The evaluation process could be extended to labeled data, allowing for the computation of standard IR metrics such as Precision@k, Recall@k, and nDCG to quantitatively measure retrieval effectiveness.

From a production-readiness standpoint, the system already implements a modular pipeline that is easy to adapt or extend. It supports local model loading and fully offline inference, which is important for cost control and future compliance with sensitive documentation constraints. The indexing step is efficient, and the hybrid retrieval pipeline combining dense and sparse retrieval followed by reranking is a robust foundation.

However, a few components would need to be implemented to move this from a proof-of-concept to a production-grade system. Currently, there is no web interface or API, which could be added using Docker + FastAPI to enable easy integration with frontend tools. There is also no continuous ingestion pipeline to handle documentation updates, meaning that the index would need to be manually rebuilt when documents change.

--- 

## Updating the Knowledge Base

The system can be re-indexed periodically using the same build_index.py script. Future iterations could support real-time indexing and invalidation of outdated chunks.

---

## Tests

Basic tests are included in the tests/ folder to validate key components of the system such as the generator pipeline.

Run tests using:
```bash
pytest tests/
```

---

## Results for suggested queries

You can see the answer and play with the Q&A assitent at **src/docs_fast_searcher.ipynb**.

- **What is SageMaker?**
```text
Token count: 1268

Generating answer...

Answer:

Amazon SageMaker is a fully managed service that enables developers and data scientists to build, train, and deploy machine learning models. It provides integrated Jupyter authoring notebook instances for easy access to data sources and eliminates the need to manage servers.


Source for the answer: examples-sagemaker.md


Other possible relevant sources for further reading: integrating-sagemaker.md, sagemaker-projects-whatis.md, kubernetes-sagemaker-jobs.md, sagemaker-projects.md
```

- **What are all AWS regions where SageMaker is available?**
```text
Token count: 1791

Generating answer...

Answer:

SageMaker is available in all supported AWS regions except Asia Pacific (Jakarta), Africa (Cape Town), Middle East (UAE), Asia Pacific (Hyderabad), Asia Pacific (Osaka), Asia Pacific (Melbourne), Europe (Milan), AWS GovCloud (US-East), Europe (Spain), China (Beijing), China (Ningxia), and Europe (Zurich) Region.


Source for the answer: sagemaker-notebook-no-direct-internet-access.md


Other possible relevant sources for further reading: sagemaker-notebook-instance-inside-vpc.md, sagemaker-compliance.md, aws-properties-sagemaker-model-containerdefinition.md, sagemaker-projects-whatis.md
```

- **How to check if an endpoint is KMS encrypted?**
```text
Token count: 2011

Generating answer...

Answer:

You can check the compliance of an Amazon SageMaker endpoint configuration regarding KMS encryption using AWS Config rules such as 'sagemaker-endpoint-configuration-kms-key-configured'. If the rule returns NON_COMPLIANT, then the KMS key is not configured for the endpoint configuration.


Source for the answer: sagemaker-roles.md


Other possible relevant sources for further reading: sagemaker-endpoint-configuration-kms-key-configured.md, aws-properties-sagemaker-featuregroup-onlinestoreconfig.md, aws-properties-sagemaker-modelpackage-transformresources.md, kubernetes-sagemaker-components-tutorials.md
```

- **What are SageMaker Geospatial capabilities?**
```text
Token count: 1355

Generating answer...

Answer:

SageMaker Geospatial capabilities are features of Amazon SageMaker that perform geospatial operations on your behalf using the AWS hardware managed by SageMaker. They can only perform operations that the user permits and require an execution role with the appropriate permissions to access AWS resources.


Source for the answer: sagemaker-geospatial-roles.md


Other possible relevant sources for further reading: sagemaker-geospatial-roles.md, integrating-sagemaker.md, examples-sagemaker.md, sagemaker-projects-whatis.md
```

You can see that the answers are quite good and are also well related to the documentation. For sure, with better models the whole pipeline would be better, from chunking to generating the answer. But anyway I would say that this POC accomplished its objectives by generating good enough answers in a reasonable time even running in my local machine (macbook air M4 16gb RAM 256 SSD). Being more specific it took me for each query something between 10 and 40 seconds to have an answer. 

---

Any questions feel free to reach out or make any PRs!
Thank you!
**André Costa Werneck**
27/06/2025 