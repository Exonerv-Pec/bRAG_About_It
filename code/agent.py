# code/agent.py — Day 6: agent behavior, error handling, logging
import logging
from pathlib import Path
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# --- Paths --------------------------------------------------------------
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
LOG_PATH = Path(__file__).resolve().parent.parent / "rag.log"

# --- Logging setup ------------------------------------------------------
# Logs to both the console and a file so every decision is visible
# while running AND reviewable afterward.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("rag.agent")

# --- Constants ----------------------------------------------------------
TOP_K = 3
# If the best similarity score is below this, the query is considered
# "weak" and will be rewritten before retrying. 0.30 was chosen based
# on the Day 5 eval: Get Out's year query scored 0.24 at the top —
# well below anything that produced a correct answer.
LOW_CONFIDENCE_THRESHOLD = 0.33

# --- Models & DB --------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"},
)
log.info(f"Chroma loaded: {collection.count()} chunks at {CHROMA_PATH}")

# --- Prompts ------------------------------------------------------------
CHITCHAT_PROMPT = """Does this message need a movie database lookup, or is it just conversation?
Reply with one word: LOOKUP or CHITCHAT.

Message: {message}"""

REWRITE_PROMPT = """The following search query returned poor results from a movie database.
Rewrite it to be more specific and descriptive so it matches movie plot descriptions better.
Return only the rewritten query, nothing else.

Original query: {query}
Rewritten query:"""

GROUNDED_PROMPT = """You are answering questions using ONLY the movie context provided below. \
If the answer isn't in the context, say "I don't have enough information in my corpus to answer that" \
instead of guessing or using outside knowledge.

Context:
{context}

Question: {question}
Answer:"""

CHITCHAT_REPLY_PROMPT = """You are a helpful movie assistant. Answer this conversational message briefly.
You have access to a small database of 8 movies: Inception, The Shawshank Redemption,
Mad Max: Fury Road, The Grand Budapest Hotel, Get Out, Interstellar, La La Land, and Parasite.

Message: {message}"""


# --- Helper: safe LLM call ----------------------------------------------
def llm(prompt: str, context: str = "LLM call") -> str:
    """
    Wraps every Ollama call with error handling and logging.
    Returns an empty string on failure so callers can decide what to do,
    rather than letting an exception bubble up and crash the whole session.
    """
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.warning(f"{context} failed: {e}")
        return ""


# --- Agent decisions ----------------------------------------------------
def route(question: str) -> str:
    """
    Decision 1: should this question trigger a database lookup at all,
    or is it just conversation? Greetings, meta-questions ("what can you
    do?"), and thanks shouldn't search the vector store.
    """
    prompt = CHITCHAT_PROMPT.format(message=question)
    verdict = llm(prompt, context="router").upper()
    decision = "LOOKUP" if "LOOKUP" in verdict else "CHITCHAT"
    log.info(f"Router: {decision!r} for {question!r}")
    return decision


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-k chunks from Chroma."""
    try:
        query_vec = model.encode(query).tolist()
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
                "similarity": round(1 - distance, 3),
            })
        return chunks
    except Exception as e:
        log.warning(f"Retrieval failed: {e}")
        return []


def maybe_rewrite(query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Decision 2: if the best similarity score is below the threshold,
    ask the LLM to rewrite the query and retry once.
    This directly addresses the Day 5 finding where short, generic
    queries (e.g. 'What year did Get Out come out?') failed to retrieve
    the right chunk.
    """
    if not chunks:
        return query, chunks

    top_score = chunks[0]["similarity"]
    if top_score >= LOW_CONFIDENCE_THRESHOLD:
        log.info(f"Retrieval confident (top={top_score}), no rewrite needed")
        return query, chunks

    log.info(f"Low confidence (top={top_score} < {LOW_CONFIDENCE_THRESHOLD}), rewriting query")
    rewritten = llm(REWRITE_PROMPT.format(query=query), context="query rewrite")

    if not rewritten:
        log.warning("Rewrite returned empty — keeping original query")
        return query, chunks

    log.info(f"Rewritten query: {rewritten!r}")
    new_chunks = retrieve(rewritten)

    if new_chunks and new_chunks[0]["similarity"] > top_score:
        log.info(f"Rewrite improved score: {top_score} → {new_chunks[0]['similarity']}")
        return rewritten, new_chunks

    log.info(f"Rewrite did not improve score ({new_chunks[0]['similarity'] if new_chunks else 'n/a'} vs {top_score}), keeping original")
    return query, chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source']} ({c['year']}, {c['genre']})] {c['text']}" for c in chunks
    )
    return GROUNDED_PROMPT.format(context=context, question=question)


# --- Main agent loop ----------------------------------------------------
def answer(question: str) -> str:
    """
    Full agent pipeline with two runtime decisions:
    1. Route: lookup vs chitchat
    2. Rewrite: retry with a better query if confidence is low
    """
    log.info(f"--- New question: {question!r}")

    # Decision 1: is a database lookup even needed?
    decision = route(question)

    if decision == "CHITCHAT":
        reply = llm(CHITCHAT_REPLY_PROMPT.format(message=question), context="chitchat reply")
        return reply or "I'm a movie assistant — ask me about any of the 8 films in my database!"

    # Decision 2: retrieve, then rewrite if confidence is low
    chunks = retrieve(question)
    if not chunks:
        return "I couldn't retrieve any context. Please check that the database is populated."

    log.info(f"Initial retrieval: top chunk [{chunks[0]['source']}] sim={chunks[0]['similarity']}")
    query, chunks = maybe_rewrite(question, chunks)

    log.info(f"Generating answer from {len(chunks)} chunks")
    prompt = build_prompt(question, chunks)
    result = llm(prompt, context="generation")
    return result or "Something went wrong generating the answer — please try again."


if __name__ == "__main__":
    print("Movie agent ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print(f"\nAgent: {answer(question)}\n")