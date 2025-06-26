from dense_embedder import DenseEmbedder
from vector_store import VectorStore
from sparse_embedder import BM25Retriever

# Step 1: Embed your test query
embedder = DenseEmbedder()
query = "How to check if an endpoint is KMS encrypted?"
query_vec = embedder.encode([query])

# Step 2: Load the vector store
store = VectorStore(dim=query_vec.shape[1])
store.load("data")

# Step 3: Retrieve results
results = store.search(query_vec, top_k=5)

# Step 4: Print results
for i, (idx, score, meta) in enumerate(results):
    print(f"\n🔎 Result {i+1} — Score: {score:.4f}")
    print(meta["text"])

# --- BM25 Retrieval ---
print("\n📚 Running BM25 Retrieval...")
bm25 = BM25Retriever()
bm25.load("data")

bm25_results = bm25.search(query, top_k=5)
for i, (score, meta) in enumerate(bm25_results):
    print(f"\n🔎 [BM25] Result {i+1} — Score: {score:.4f}")
    print(meta["text"])