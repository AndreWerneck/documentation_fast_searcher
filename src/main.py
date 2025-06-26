from sentence_transformers.cross_encoder import CrossEncoder
from vector_store import VectorStore
from dense_embedder import DenseEmbedder
from sparse_embedder import BM25Retriever
from generator import LLMGenerator
from config import MAX_TOKENS

def format_prompt(query: str, contexts: list[dict]) -> str:
    """Constructs the prompt with retrieved contexts and user query."""
    context_str = "\n\n".join(
        f"Chunk {i+1}:\n{chunk['text'].strip()}"
        for i, chunk in enumerate(contexts)
    )

    return f"""You are a helpful assistant answering questions about AWS SageMaker documentation. Base your answers on the provided context.
If the context does not contain enough information, you can answer with "I don't know".
Lastly, if you are sure about the answer, you can answer even if the information is not in the context.
There is always just one question, so do not answer multiple questions at once.
Give concise and accurate answers.

Context:
{context_str}

Question: {query}
Answer:"""

def main():
    # Initialize components
    dense_embedder = DenseEmbedder()
    sparse_embedder = BM25Retriever()
    generator = LLMGenerator()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    # --- Step 1: Get user query ---
    query = input("🔍 Ask your question: ")

    # --- Step 2: Retrieve from dense store ---
    query_vec = dense_embedder.encode([query])
    store = VectorStore(dim=query_vec.shape[1])
    store.load("data")
    dense_results = store.search(query_vec, top_k=5)
    dense_chunks = [meta for _, _, meta in dense_results]

    # --- Step 3: Retrieve from sparse BM25 ---
    sparse_embedder.load("data")
    sparse_results = sparse_embedder.search(query, top_k=5)
    sparse_chunks = [meta for _, meta in sparse_results]

    # --- Step 4: Merge and deduplicate ---
    all_candidates = {doc["id"]: doc for doc in dense_chunks + sparse_chunks}
    candidate_chunks = list(all_candidates.values())

    # --- Step 5: Rerank ---
    rerank_inputs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = reranker.predict(rerank_inputs)

    for chunk, score in zip(candidate_chunks, scores):
        chunk["score"] = float(score)

    top_reranked = sorted(candidate_chunks, key=lambda x: x["score"], reverse=True)[:5]

    # --- Step 6: Format and generate response ---
    prompt = format_prompt(query, top_reranked)
    print("🧮 Token count:", generator.count_tokens(prompt))

    print("\n🧠 Generating answer...\n")
    answer = generator.llmgenerate(prompt=prompt, max_tokens=MAX_TOKENS)

    print("✅ Answer:\n")
    print(answer)
    print('\n')
    print(f'Source for the answer: {top_reranked[0]["source"]}')
    print('\n')
    print(f"Other possible relevant sources for further reading: {', '.join(chunk['source'] for chunk in top_reranked[1:])}")

if __name__ == "__main__":
    main()