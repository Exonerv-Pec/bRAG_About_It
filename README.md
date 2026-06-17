# bRAG About It

A small command-line tool that answers questions over a set of documents I choose — built from scratch over a week to show Retrieval Augmented Generation end-to-end.

## Status: Day 2 of 7 — embeddings + semantic search

🟩🟩⬜⬜⬜⬜⬜ 2/7 days (29%)

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
   First run of the search script also downloads the embedding model (~80MB) from Hugging Face — needs internet once, then it's cached locally.

## Usage

Run the first LLM test:
```
python code\hello_ollama.py
```

Run semantic search over the movie corpus:
```
python code\movie_search.py
```

## Progress log

- **Day 1** — repo set up, Ollama running locally, first script-generated LLM response.
- **Day 2** — built semantic search over an 8-movie corpus using `sentence-transformers` + manual cosine similarity. Key findings:
  - Similarity scores from this model cluster in a narrow band (roughly 0.0-0.4) for everyday sentences — a "high" score is nowhere near 1.0, and a small numeric gap between results can still reflect a real ranking difference.
  - Short queries (2-5 words) are unstable: rewording "Saves Humanity" to "A person saves the future" flipped Interstellar from the top result (0.296) to nearly last (0.173). With so few words, each one carries outsized weight.
  - Longer, descriptive queries are far more reliable: "space exploration to save humanity" scored Interstellar at 0.544 with the next-best result under 0.16 — a clean, confident match. Same result rewording La La Land's plot with zero shared vocabulary (0.589 vs. 0.203 next-best).
  - Takeaway for later: query phrasing matters a lot for retrieval quality — relevant for Day 4's prompt engineering and Day 6's idea of an agent rewriting a weak query before retrying.

## What's next

- Day 3 — chunking + Chroma vector store
- Day 4 — retrieval + generation + prompt engineering
- Day 5 — evaluation
- Day 6 — agent behavior
- Day 7 — polish + final write-up
- Stretch (Day 8) — pluggable LLM backend: let someone clone this repo and use their own Claude or Gemini API key instead of Ollama, via an env variable + a small provider wrapper