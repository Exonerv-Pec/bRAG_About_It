# code/ingest.py — Day 3: chunk + embed + store in Chroma
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movies.json"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"

CHUNK_SIZE = 300       # characters per chunk
OVERLAP_RATIO = 0.15   # ~15% overlap between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO):
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    print(f"Reading data from: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        movies = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Writing Chroma DB to: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="movies",
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for movie in movies:
        chunks = chunk_text(movie["synopsis"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{movie['title']}_chunk{i}")
            texts.append(chunk)
            metadatas.append({
                "source": movie["title"],
                "chunk_index": i,
                "year": movie["year"],
                "genre": movie["genre"],
            })

    print(f"Chunked {len(movies)} movies into {len(texts)} chunks.")
    print("Embedding...")
    embeddings = model.encode(texts, show_progress_bar=True)

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks at {CHROMA_PATH}")


if __name__ == "__main__":
    main()