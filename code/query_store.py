# code/query_store.py — Day 3: query the persisted store directly, no LLM yet
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"},
)

print(f"Chroma path: {CHROMA_PATH}  |  chunks stored: {collection.count()}")


def search(query, top_k=5):
    query_vec = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    print(f"\nQuery: {query!r}")
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - distance
        print(f"  sim={similarity:.3f}  [{meta['source']} chunk {meta['chunk_index']}]  {doc[:90]}...")


if __name__ == "__main__":
    q = input("Search query: ")
    search(q)