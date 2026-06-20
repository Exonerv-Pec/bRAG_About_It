# code/query_store.py — Day 3: query the persisted store directly, no LLM yet
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"},
)


def search(query, top_k=5):
    query_vec = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    print(f"\nQuery: {query!r}")
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma returns cosine DISTANCE (lower = more similar) — the
        # opposite direction from Day 2's similarity (higher = more similar).
        # This converts it back to the scale you already know how to read.
        similarity = 1 - distance
        print(f"  sim={similarity:.3f}  [{meta['source']} chunk {meta['chunk_index']}]  {doc[:90]}...")


if __name__ == "__main__":
    q = input("Search query: ")
    search(q)