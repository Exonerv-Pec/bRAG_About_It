# code/ingest.py — Day 3: chunk + embed + store in Chroma
import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNK_SIZE = 300       # characters per chunk
OVERLAP_RATIO = 0.15   # ~15% overlap between consecutive chunks

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movies.json"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO):
    """
    Split text into overlapping chunks. The overlap exists so a sentence
    that would otherwise get cut across two chunks still has some
    surrounding context on at least one side.
    """
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
    with open(DATA_PATH, encoding="utf-8") as f:
        movies = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Persisted client — this is the part that didn't exist on Day 2.
    # Anything written here survives between runs; you only need to
    # re-embed when the source data actually changes.
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="movies",
        metadata={"hnsw:space": "cosine"},  # match the similarity metric from Day 2
    )

    ids, texts, metadatas = [], [], []
    for movie in movies:
        chunks = chunk_text(movie["synopsis"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{movie['title']}_chunk{i}")
            texts.append(chunk)
            # source + chunk_index let you trace any result back to exactly
            # which document and which slice of it it came from. year/genre
            # are carried along too — not embedded, just attached, ready for
            # metadata filtering later.
            metadatas.append({
                "source": movie["title"],
                "chunk_index": i,
                "year": movie["year"],
                "genre": movie["genre"],
            })

    print(f"Chunked {len(movies)} movies into {len(texts)} chunks.")
    print("Embedding...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # upsert so re-running this script after editing movies.json doesn't
    # error out on duplicate ids — it just overwrites them
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ./chroma_db")


if __name__ == "__main__":
    main()