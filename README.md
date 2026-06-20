# bRAG About It

A small command-line tool that answers questions over a set of documents I choose — built from scratch over a week to show Retrieval Augmented Generation end-to-end.

## Status: Day 3 of 7 — chunking + persisted vector store

🟩🟩🟩⬜⬜⬜⬜ 3/7 days (43%)

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
```

## Progress log

**Day 1** — Repo set up, Ollama running locally, got a real response back from a script-generated prompt.

**Day 2** — Semantic search over 8 movie descriptions using sentence-transformers and manual cosine similarity. The interesting part wasn't getting it working, it was how unstable short queries turned out to be. Rewording "Saves Humanity" to "A person saves the future" flipped Interstellar from a 0.296 top score to nearly last place at 0.173 — same intent, completely different ranking, just because there's so little text for the model to average over. Longer, more descriptive queries were way more consistent (0.544 for a 5-word description with the next result under 0.16).

**Day 3** — Moved from a flat Python list to an actual persisted store. Expanded each movie into a full synopsis paragraph, chunked them (300 chars, ~15% overlap), embedded the chunks, and stored everything in Chroma with metadata (source title, chunk index, year, genre). Querying now goes straight against the disk-backed collection instead of re-embedding everything on every run.

Some results worth noting once it was working:
- *"Max Rockatansky finds himself captured by the War Boys"* → Mad Max chunk 0 at 0.733, by far the most confident match I've seen so far — makes sense, it's basically quoting the synopsis back.
- *"I want to watch a movie where actress works as barista!"* → La La Land's barista chunk came back top at 0.417 despite the casual phrasing and exclamation mark.
- *"Yo! I want to watch a movie about a banker!"* → this one was the most interesting. The Grand Budapest Hotel narrowly outscored Shawshank Redemption (0.336 vs 0.319) — even though Shawshank's synopsis literally contains the word "banker." The model is matching on broader plot vibe more than exact word hits, which cuts both ways.
- I also noticed the same movie can show up twice in one result list since each synopsis is now split into several chunks. One of Shawshank's chunks that scored 0.299 starts mid-sentence ("le long-term plan culminate...") because the boundary cut it off mid-word. It still ranked respectably on topic alone, but it's not the chunk I'd actually want surfaced as the answer. Something to look at more carefully once eval scoring is in place on Day 5 — possibly worth experimenting with different chunk sizes too.

## What's next

- Day 4 — retrieval + generation + prompt engineering
- Day 5 — evaluation
- Day 6 — agent behavior
- Day 7 — polish + final write-up
- Stretch (Day 8) — pluggable LLM backend so someone cloning this repo can use their own Claude or Gemini key instead of Ollama
