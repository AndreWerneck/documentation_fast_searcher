from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from preprocessing import DocumentChunk
import numpy as np
import json

class DenseEmbedder:
    """
    A class to embed text chunks using a pre-trained SentenceTransformer model.
    It encodes text into dense vector representations and saves them along with metadata.
    """
    def __init__(self, model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1'):
        """ Initializes the embedder with a specified SentenceTransformer model.
        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into embeddings.
        Args:
            texts (List[str]): List of text strings to encode.
            Returns:
                np.ndarray: Array of encoded text embeddings.
        """
        return self.model.encode(texts) #show_progress_bar=True)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> Tuple[np.ndarray, List[dict]]:
        """
        Embeds a list of DocumentChunk objects and returns their embeddings and metadata.
        Args:
            chunks (List[DocumentChunk]): List of DocumentChunk objects to embed.
        Returns:
            Tuple[np.ndarray, List[dict]]: A tuple containing the embeddings and metadata.
        """
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

    def save(self, embeddings: np.ndarray, metadata: List[dict], output_dir: str = "../data"):
        """ Saves the embeddings and metadata to specified output directory.
        Args:
            embeddings (np.ndarray): Array of embeddings to save.
            metadata (List[dict]): List of metadata dictionaries to save.
            output_dir (str): Directory to save the files.
        """
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
    embedder = DenseEmbedder()
    embeddings, metadata = embedder.embed_chunks(chunks)
    embedder.save(embeddings, metadata)

    print(f"Embedded {len(embeddings)} chunks and saved to 'data/'")