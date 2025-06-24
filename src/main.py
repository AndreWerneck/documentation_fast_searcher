
import numpy as np
import json
from vector_store import VectorStore
from embedder import Embedder
from generator import LLMGenerator
from config import MAX_TOKENS

def format_prompt(query: str, contexts: list[dict]) -> str:
    """Constructs the prompt with retrieved contexts and user query."""
    context_str = "\n\n".join(
        f"Chunk {i+1}:\n{chunk['text'].strip()}"
        for i, chunk in enumerate(contexts)
    )

    prompt = f"""You are a helpful assistant answering questions about AWS SageMaker documentation. Base your answers on the provided context.
    If the context does not contain enough information, you can answer with "I don't know". If the context is not relevant, you can also answer with "context not relevant for this query".
    Lastly, if you are sure about the answer, you can answer even if the information is not in the context.
    There is always just one question, so do not answer multiple questions at once.
    Give concise and accurate answers.

Context:
{context_str}

Question: {query}
Answer:"""
    return prompt

def main():
    # Load embedder and generator
    embedder = Embedder()
    generator = LLMGenerator()

    # Get user query
    query = input("🔍 Ask your question: ")

    # Embed and search
    query_vec = embedder.encode([query])
    # Load vector store
    store = VectorStore(dim=query_vec.shape[1])
    store.load("data")
    top_results = store.search(query_vec, top_k=5)

    # Format context
    context_chunks = [meta for _, _, meta in top_results]
    prompt = format_prompt(query, context_chunks)

    print("Token count:", generator.count_tokens(prompt))

    # Generate response
    print("\n🧠 Generating answer...\n")
    answer = generator.llmgenerate(prompt=prompt, max_tokens=MAX_TOKENS)
    print("✅ Answer:\n")
    print(answer)

if __name__ == "__main__":
    main()