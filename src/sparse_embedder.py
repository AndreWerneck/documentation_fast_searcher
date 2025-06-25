import json
import os
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from preprocessing import DocumentChunk

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.metadata = []

    def build(self, chunks: List[DocumentChunk]):
        self.documents = [word_tokenize(chunk.text.lower()) for chunk in chunks]
        self.metadata = [
            {
                "id": chunk.id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.documents)

    def save(self, output_dir: str = "data"):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "bm25_docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load(self, index_dir: str = "data"):
        with open(os.path.join(index_dir, "bm25_docs.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.documents = [word_tokenize(doc["text"].lower()) for doc in self.metadata]
        self.bm25 = BM25Okapi(self.documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        if self.bm25 is None:
            raise ValueError("BM25 index not built or loaded.")

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(scores[i], self.metadata[i]) for i in top_indices]