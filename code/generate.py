# code/generate.py — Day 4: retrieval + generation + prompt engineering
from pathlib import Path
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
TOP_K = 3

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"},
)

print(f"Chroma path: {CHROMA_PATH}  |  chunks stored: {collection.count()}")
if collection.count() == 0:
    print("WARNING: the collection is empty at this path.")
    print("Run ingest.py first, and check it reports writing to this exact path.")

# --- Prompt templates ---------------------------------------------------

# NAIVE: just hands the model context and a question, no instructions
# about staying grounded. This is the "before" version of today's
# experiment -- worth running once on its own so you can see what
# happens when nobody tells the model to stick to the context.
PROMPT_NAIVE = """Context:
{context}

Question: {question}
Answer:"""

# GROUNDED: explicitly tells the model to answer only from the context
# and to admit when it can't. This single instruction is most of what
# separates a grounded RAG answer from a hallucinated one -- though as
# you've already seen, "most of" is not "all of".
PROMPT_GROUNDED = """You are answering questions using ONLY the movie context provided below. \
If the answer isn't in the context, say "I don't have enough information in my corpus to answer that" \
instead of guessing or using outside knowledge.

Context:
{context}

Question: {question}
Answer:"""

# Flip this to False, rerun, and compare against a question you know
# isn't answerable from the context.
USE_GROUNDED_PROMPT = True


def retrieve(question, top_k=TOP_K):
    query_vec = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "year": meta.get("year"),
            "genre": meta.get("genre"),
            "similarity": 1 - distance,
        })
    return chunks


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{c['source']} ({c['year']}, {c['genre']})] {c['text']}" for c in chunks
    )
    template = PROMPT_GROUNDED if USE_GROUNDED_PROMPT else PROMPT_NAIVE
    return template.format(context=context, question=question)


def answer(question, top_k=TOP_K, verbose=True):
    chunks = retrieve(question, top_k)
    prompt = build_prompt(question, chunks)

    if verbose:
        print("\n--- Retrieved chunks ---")
        if not chunks:
            print("  (none returned -- collection may be empty, or path mismatch)")
        for c in chunks:
            print(f"  sim={c['similarity']:.3f}  [{c['source']}]")
        print("\n--- Prompt sent to the LLM ---")
        print(prompt)

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = input("\nAsk a question about the movies: ")
    result = answer(question)  # all verbose printing happens during this call
    print("\n--- Answer ---")
    print(result)