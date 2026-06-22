# bRAG About It

A small command-line tool that answers questions over a set of documents I choose — built from scratch over a week to show Retrieval Augmented Generation end-to-end.

## Status: Day 5 of 7 — evaluation

🟩🟩🟩🟩🟩⬜⬜ 5/7 days (71%)

## Stack

| Piece | Choice | Why |
|---|---|---|
| LLM | Ollama (`llama3.2`) | Local, free, no API key needed |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free, no rate limits |
| Vector store | Chroma | Pure Python, persists to disk |

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
   ```
   ollama pull llama3.2
   ```
2. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
   First run also downloads the embedding model (~80MB) from Hugging Face. Needs internet once, then it's cached locally.

## Usage

```
python code\hello_ollama.py      # sanity check the LLM connection
python code\movie_search.py      # in-memory semantic search (Day 2)
python code\ingest.py            # chunk + embed + store in Chroma (Day 3)
python code\query_store.py       # query the persisted store directly
python code\generate.py          # full RAG pipeline: retrieve + prompt + generate (Day 4)
python code\evaluate.py          # run the 12-question eval harness (Day 5)
```

## Evaluation results (Day 5)

| Metric | Score | Notes |
|---|---|---|
| Retrieval recall@3 | 10/11 = **91%** | 1 miss: short generic query for Get Out retrieved wrong chunks |
| Refusal detection | **100%** | Reliable — caught by code, not by the LLM judge |
| GROUNDED/UNGROUNDED | **unreliable** | `llama3.2` too small for consistent faithfulness scoring |

---

## Progress log

### Day 1 — Project setup + first LLM call

Repo set up, Ollama running locally, first script-generated LLM response working.

<details>
<summary>Full notes</summary>

Set up the GitHub repo, virtual environment, and `.gitignore`. Installed Ollama on Windows, pulled `llama3.2`, and wrote a minimal script that sends a prompt and prints the response. Hit the usual first-time Windows hiccup where PowerShell blocked the venv activation script — fixed with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Also set up the `.env` / `.env.example` pattern for API key safety, even though Ollama doesn't need one, since most real projects eventually touch at least one key.

</details>

---

### Day 2 — Embeddings + semantic search

Built semantic search over 8 movie descriptions. The key finding was how unstable short queries are — two words can flip the top result entirely.

<details>
<summary>Full notes</summary>

Installed `sentence-transformers` and used `all-MiniLM-L6-v2` to embed a hand-built movie corpus, then computed cosine similarity manually against query vectors.

The most interesting finding wasn't getting it to work — it was how much query length matters. Rewording "Saves Humanity" to "A person saves the future" flipped Interstellar from the top result (0.296) to nearly last (0.173), even though the meaning is identical. With only 2 words, each individual word carries a disproportionate share of the resulting vector, so small wording changes cause big ranking swings. Longer, more descriptive queries were dramatically more consistent: "space exploration to save humanity" scored Interstellar at 0.544 with the next result under 0.16 — a clean, confident match.

Also confirmed the "real test" from the plan: querying "Two young performers in California chase their career ambitions and develop a romance" returned La La Land at 0.589 despite sharing almost no vocabulary with the actual synopsis. That gap between zero word overlap and a correct top result is what semantic search is for.

Model used: `all-MiniLM-L6-v2` (384-dimensional embeddings, ~80MB). The narrow score range (most results clustering between 0.0–0.5 rather than spanning 0–1) is a known characteristic of small embedding models — the ranking is still meaningful, but the absolute numbers aren't confidence scores.

</details>

---

### Day 3 — Chunking + persisted vector store

Moved from a flat Python list to a real Chroma database that persists between runs. 8 movies → 26 chunks stored on disk.

<details>
<summary>Full notes</summary>

Expanded each movie from a one-line description to a full synopsis paragraph (necessary to give the chunker something real to work with), then built a chunking function: 300 characters per chunk, ~15% overlap between consecutive chunks. The overlap doesn't link chunks together — it just copies the tail of one chunk to the head of the next, so a sentence that falls on a boundary appears complete in at least one of the two chunks.

Stored everything in a Chroma `PersistentClient` collection with metadata attached to each chunk: source title, chunk index, year, genre. Year and genre aren't embedded — they ride along as metadata for filtering, not for similarity matching.

Hit a path bug on the first run: PyCharm's default working directory is the script's own folder (`code/`), so `open("data/movies.json")` was looking for `code/data/movies.json`. Fixed by computing all paths off `Path(__file__).resolve().parent.parent` so every script finds its files regardless of where it's launched from.

Some results from manual testing once it was working:
- *"Max Rockatansky finds himself captured by the War Boys"* → Mad Max chunk 0 at 0.733 — by far the most confident match seen all week.
- *"I want to watch a movie where actress works as barista!"* → La La Land top at 0.417 despite casual phrasing and punctuation.
- *"Yo! I want to watch a movie about a banker!"* → Grand Budapest narrowly beat Shawshank (0.336 vs 0.319) even though "banker" appears verbatim in Shawshank's synopsis. The model is matching on overall plot vibe, not keyword presence.
- One Shawshank chunk starts mid-sentence ("...le long-term plan culminate...") because the chunk boundary fell mid-word. It kept appearing as noise in unrelated queries throughout Days 4 and 5.

</details>

---

### Day 4 — Retrieval + generation + prompt engineering

Closed the loop into a full RAG pipeline. Found that a grounding instruction meaningfully reduces hallucination but doesn't eliminate it — especially when the model is confident it already knows the answer.

<details>
<summary>Full notes</summary>

Wired the full path: `retrieve` → `build_prompt` → call Ollama → print answer. The prompt template uses `.format()` rather than an f-string because `context` and `question` don't exist at the time the template is defined — they're substituted later inside `build_prompt()` when a real question comes in.

Ran the grounded vs naive comparison. The naive version produced the cleanest hallucination of the week: asked "Do you know any movie about a barista?", it had La La Land's synopsis sitting right there in the context and never mentioned it. Instead it invented a movie called "Barista" (2023) with a fabricated director, and cited "Like Water for Chocolate" as if it came from the provided context, which it didn't.

The grounded version went 7 for 7 on a first pass — until one more test. Mad Max's synopsis uses only character names (Max, Furiosa, Immortan Joe), never actor names. Asked "Who are the main actors in Mad Max: Fury Road?" with the correct synopsis retrieved and the grounding instruction active, the model answered with a full five-person cast list — all factually accurate, none of it in the context. The instruction was bypassed entirely because the model was confident in its own training data.

The takeaway: a grounding instruction is a strong nudge, not a hard guarantee. It breaks down specifically when the right document is retrieved but the one specific fact being asked about isn't in it — the most realistic failure mode in production, because it's invisible unless you actually check the context.

</details>

---

### Day 5 — Evaluation

Built a 12-question harness with separate retrieval (recall@3) and generation (faithfulness) scores. Retrieval works well. The automated judge turned into a whole investigation of its own.

<details>
<summary>Full notes</summary>

Built `evaluate.py` with a 12-question test set drawn from real findings across Days 2–4. Each question has an `expected_source` (which movie's chunk should be retrieved) and an `answerable` flag (whether the corpus actually contains the specific fact, not just the right document). That distinction matters: Mad Max's synopsis is the right document for actor questions, but it doesn't contain actor names — so the correct answer is a refusal, even when retrieval succeeds.

**Retrieval recall@3: 91% (10/11)** — genuinely solid. The one miss (Get Out's year) is consistent with the Day 2 finding: short, generic queries don't give the embedding enough signal to find the right chunk.

The generation scoring went through three iterations and turned into the most honest finding of the week.

**Iteration 1** — original judge prompt, no label ordering fix. The substring bug meant `"GROUNDED"` matched inside `"UNGROUNDED"` first, so every verdict came back GROUNDED regardless of actual content. Fixed by reordering the label check to test `"UNGROUNDED"` before `"GROUNDED"`.

**Iteration 2** — substring fix applied. Refusals were still misclassified (the LLM returned GROUNDED on clear "I don't have enough information" answers). Fixed by moving refusal detection out of the LLM entirely and into a simple string match in Python — much more reliable since the refusal phrasing is consistent.

**Iteration 3** — hybrid approach: refusals detected by code, GROUNDED/UNGROUNDED by the LLM. Refusal detection now works perfectly. But the LLM judge is still wrong in the other direction — it marks short, clearly grounded answers like "The Kim family" and "M. Gustave" as UNGROUNDED, and even returns misspelled labels like "UNGROUNDENED." and "UNGROUNDEN." showing it's not confidently following the instruction at all.

The conclusion: `llama3.2` is too small to be a reliable faithfulness judge. This is a known limitation in the field — LLM-as-judge works well with large, instruction-following models (GPT-4, Claude Opus) but degrades significantly with smaller local models. The refusal check stays in code. The GROUNDED/UNGROUNDED scoring is flagged as unreliable in this setup, and would need a larger model or a rule-based approach to be production-worthy.

The real lesson isn't that the eval harness is broken — it's that evaluation itself is a hard problem, and the tool you use to measure quality has its own quality problem. That applies to any automated eval pipeline, not just this one.

</details>

---

## What's next

- Day 6 — agent behavior
- Day 7 — polish + final write-up
- Stretch (Day 8) — pluggable LLM backend so someone cloning this repo can use their own Claude or Gemini key instead of Ollama