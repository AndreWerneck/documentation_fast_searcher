import faiss
import numpy as np
import json
from typing import List, Tuple

class VectorStore:
    """A class to manage a vector store using FAISS for efficient similarity search.
    """
    def __init__(self, dim: int, index_path: str = None):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []
        self.index_path = index_path

    def add(self, vectors: np.ndarray, metadata: List[dict]):
        """Adds vectors and their corresponding metadata to the FAISS index.
        Args:
            vectors (np.ndarray): Array of shape (n_samples, dim) containing the vectors to add.
            metadata (List[dict]): List of metadata dictionaries corresponding to each vector.
        """
        faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, dict]]:
        """Searches for the top_k most similar vectors to the query vector.
        Args:
            query_vector (np.ndarray): Array of shape (1, dim) containing the query vector.
            top_k (int): The number of top similar vectors to retrieve.
        Returns:
            List[Tuple[int, float, dict]]: A list of tuples containing the index, distance, and metadata of the top_k similar vectors.
        """
        faiss.normalize_L2(query_vector)  # Normalize query vector
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                results.append((idx, dist, self.metadata[idx]))
        return results

    def save(self, dir_path: str):
        """Saves the FAISS index and metadata to the specified directory.
        Args:
            dir_path (str): Directory path where the index and metadata will be saved.
        """
        faiss.write_index(self.index, f"{dir_path}/faiss.index")
        with open(f"{dir_path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load(self, dir_path: str):
        """Loads the FAISS index and metadata from the specified directory.
        """
        self.index = faiss.read_index(f"{dir_path}/faiss.index")
        with open(f"{dir_path}/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

if __name__ == "__main__":
    
    from dense_embedder import DenseEmbedder

    # Load embeddings and metadata
    vectors = np.load("../data/embeddings.npy")
    with open("../data/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Build and save index
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, metadata)
    store.save("../data")

    # Embed a test query
    embedder = DenseEmbedder()
    query = "What is SageMaker?"
    query_vec = embedder.encode([query])

    # Search
    results = store.search(query_vec, top_k=3)
    for i, (idx, dist, meta) in enumerate(results):
        print(f"\nResult {i+1} — Distance: {dist:.4f}")
        print(meta["text"])