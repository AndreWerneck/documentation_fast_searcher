import json
import numpy as np
from pathlib import Path

from preprocessing import MarkdownPreprocessor
from dense_embedder import DenseEmbedder
from sparse_embedder import BM25Retriever
from vector_store import VectorStore

OUTPUT_DIR = "data"

def run_pipeline(input_dir: str = "sagemaker_documentation"):
    """Runs the full pipeline to preprocess, embed, and index Markdown files.
    Args:
        input_dir (str): Directory containing Markdown files.
    """

    print("Step 1: Preprocessing Markdown files...")
    preprocessor = MarkdownPreprocessor(input_dir)
    chunks = preprocessor.load_all_markdown(chunking_type='semantic')
    print(f"{len(chunks)} chunks generated.")
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump(mode='json') for chunk in chunks], f, indent=2, ensure_ascii=False)

    print("Step 2: Embedding chunks...")
    embedder = DenseEmbedder()
    vectors, metadata = embedder.embed_chunks(chunks)
    print(f"Embeddings shape: {vectors.shape}")

    print("Step 3: Saving embeddings and metadata...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(f"{OUTPUT_DIR}/embeddings.npy", vectors)
    with open(f"{OUTPUT_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Step 4: Building FAISS index...")
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata)
    store.save(OUTPUT_DIR)

    print("Step 5: Building BM25 sparse index...")
    bm25 = BM25Retriever()
    bm25.build(chunks)
    bm25.save(OUTPUT_DIR)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()