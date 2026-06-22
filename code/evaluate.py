# code/evaluate.py — Day 5: evaluation harness
from pathlib import Path
import csv
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.csv"
TOP_K = 3

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name="movies",
    metadata={"hnsw:space": "cosine"},
)

print(f"Chroma path: {CHROMA_PATH}  |  chunks stored: {collection.count()}")

# --- Test set -------------------------------------------------------
# expected_source : which movie's chunk SHOULD be retrieved.
#                   None = the question is about a movie outside the
#                   corpus entirely, so retrieval has nothing to find.
# answerable      : whether the corpus, as actually embedded, contains
#                   the fact needed. True = a real answer is expected.
#                   False = "I don't have enough information..." is
#                   the CORRECT answer, even if the right document
#                   gets retrieved (e.g. Mad Max's synopsis never
#                   names actors - the doc is right, the fact isn't there).
TEST_SET = [
    {"id": "q01", "question": "What happens in Inception?",
     "expected_source": "Inception", "answerable": True},
    {"id": "q02", "question": "Who tries to escape from prison in The Shawshank Redemption?",
     "expected_source": "The Shawshank Redemption", "answerable": True},
    {"id": "q03", "question": "What family infiltrates a wealthy household?",
     "expected_source": "Parasite", "answerable": True},
    {"id": "q04", "question": "What genre is La La Land?",
     "expected_source": "La La Land", "answerable": True},
    {"id": "q05", "question": "What year did Get Out come out?",
     "expected_source": "Get Out", "answerable": True},
    {"id": "q06", "question": "How long is Interstellar?",
     "expected_source": "Interstellar", "answerable": False},
    {"id": "q07", "question": "Who are the main actors in Mad Max: Fury Road?",
     "expected_source": "Mad Max: Fury Road", "answerable": False},
    {"id": "q08", "question": "Can you tell me something about The Fifth Element?",
     "expected_source": None, "answerable": False},
    {"id": "q09", "question": "Do you know any movie about a barista?",
     "expected_source": "La La Land", "answerable": True},
    {"id": "q10", "question": "Who is the concierge at the Grand Budapest Hotel?",
     "expected_source": "The Grand Budapest Hotel", "answerable": True},
    {"id": "q11", "question": "What does Cooper do for a living before the mission in Interstellar?",
     "expected_source": "Interstellar", "answerable": True},
    {"id": "q12", "question": "What weapon does Immortan Joe use?",
     "expected_source": "Mad Max: Fury Road", "answerable": False},
]

PROMPT_GROUNDED = """You are answering questions using ONLY the movie context provided below. \
If the answer isn't in the context, say "I don't have enough information in my corpus to answer that" \
instead of guessing or using outside knowledge.

Context:
{context}

Question: {question}
Answer:"""


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
    return PROMPT_GROUNDED.format(context=context, question=question)


def generate_answer(question, chunks):
    prompt = build_prompt(question, chunks)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def judge_answer(question, chunks, answer):
    """
    LLM-as-judge: determine whether the answer is grounded, ungrounded,
    or a refusal. Uses a stepped approach: check for refusal first
    (simple string match, no LLM needed), then ask the LLM only for
    the harder GROUNDED vs UNGROUNDED distinction.
    """
    # Step 1: detect refusals with a simple string check rather than
    # asking the LLM -- the model's refusal phrasing is consistent enough
    # that this is more reliable than asking llama3.2 to classify it.
    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "not in the context",
        "cannot answer",
        "no information",
    ]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return "REFUSED"

    # Step 2: only call the LLM for answers that aren't refusals,
    # where it actually needs to compare the answer against the context.
    context = "\n\n".join(c["text"] for c in chunks)
    judge_prompt = f"""You are a strict fact-checker. Does the AI answer below contain ONLY information that appears in the context? Answer with one word.

Context:
{context}

Question: {question}
AI answer: {answer}

Rules:
- If every fact in the answer is explicitly stated in the context above, respond: GROUNDED
- If the answer contains ANY fact, name, number, or detail not present in the context above, respond: UNGROUNDED
- Do not consider whether the answer is factually true in the real world -- only whether it came from the context.

One word only (GROUNDED or UNGROUNDED):"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = response["message"]["content"].strip().upper()
    for label in ("UNGROUNDED", "GROUNDED"):
        if label in verdict:
            return label
    return verdict


def run_evaluation():
    rows = []
    hits, retrieval_total = 0, 0

    for case in TEST_SET:
        chunks = retrieve(case["question"])
        retrieved_sources = {c["source"] for c in chunks}

        # --- retrieval scoring (recall@k) ---
        if case["expected_source"] is not None:
            retrieval_total += 1
            hit = case["expected_source"] in retrieved_sources
            hits += hit
        else:
            hit = None  # not applicable - no right answer exists in the corpus

        # --- generation scoring ---
        answer = generate_answer(case["question"], chunks)
        verdict = judge_answer(case["question"], chunks, answer)

        # Correct is judged against what the model actually received, not
        # an idealized "if retrieval had worked" version - a retrieval
        # miss followed by an honest refusal is a GENERATION success,
        # even though it's a RETRIEVAL failure. Keeping these separates is
        # the whole point of today.
        if not case["answerable"]:
            correct = verdict == "REFUSED"
        elif hit:
            correct = verdict == "GROUNDED"
        else:
            correct = verdict == "REFUSED"

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "retrieved_sources": ", ".join(sorted(retrieved_sources)),
            "retrieval_hit": hit,
            "answerable": case["answerable"],
            "judge_verdict": verdict,
            "correct": correct,
            "answer": answer.replace("\n", " "),
        })
        print(f"{case['id']}: retrieval_hit={hit}  judge={verdict}  correct={correct}")

    recall_at_k = hits / retrieval_total if retrieval_total else 0
    gen_correct = sum(r["correct"] for r in rows)

    print(f"\nRecall@{TOP_K}: {hits}/{retrieval_total} = {recall_at_k:.0%}")
    print(f"Generation accuracy: {gen_correct}/{len(rows)} = {gen_correct / len(rows):.0%}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()