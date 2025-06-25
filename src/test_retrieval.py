from embedder import Embedder
from vector_store import VectorStore

# Step 1: Embed your test query
embedder = Embedder()
query = "What are SageMaker Geospatial capabilities?"
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