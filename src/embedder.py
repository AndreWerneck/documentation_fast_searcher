from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from preprocessing import DocumentChunk
import numpy as np
import json

class Embedder:
    def __init__(self, model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'): #= "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into embeddings."""
        return self.model.encode(texts) #show_progress_bar=True)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> Tuple[np.ndarray, List[dict]]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encode(texts)#, show_progress_bar=True)

        metadata = [
            {
                **chunk.metadata,
                "source": chunk.source,
                "id": chunk.id,
                "text": chunk.text
            }
            for chunk in chunks
        ]

        return np.array(embeddings), metadata

    def save(self, embeddings: np.ndarray, metadata: List[dict], output_dir: str = "data"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        np.save(f"{output_dir}/embeddings.npy", embeddings)

        with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Load preprocessed chunks
    from preprocessing import DocumentChunk, MarkdownPreprocessor

    preprocessor = MarkdownPreprocessor(input_dir="sagemaker_documentation")
    chunks = preprocessor.load_all_markdown()

    # Embed and save
    embedder = Embedder()
    embeddings, metadata = embedder.embed_chunks(chunks)
    embedder.save(embeddings, metadata)

    print(f"✅ Embedded {len(embeddings)} chunks and saved to 'data/'")