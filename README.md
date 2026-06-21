# bRAG About It

A small command-line tool that answers questions over a set of documents I choose — built from scratch over a week to show Retrieval Augmented Generation end-to-end.

## Status: Day 4 of 7 — retrieval + generation + prompt engineering

🟩🟩🟩🟩⬜⬜⬜ 4/7 days (57%)

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
```

`generate.py` has a `USE_GROUNDED_PROMPT` flag at the top — flip it to compare a naive prompt against one with explicit grounding instructions.

## Progress log

**Day 1** — Repo set up, Ollama running locally, got a real response back from a script-generated prompt.

**Day 2** — Semantic search over 8 movie descriptions using sentence-transformers and manual cosine similarity. The interesting part wasn't getting it working, it was how unstable short queries turned out to be. Rewording "Saves Humanity" to "A person saves the future" flipped Interstellar from a 0.296 top score to nearly last place at 0.173 — same intent, completely different ranking, just because there's so little text for the model to average over. Longer, more descriptive queries were way more consistent (0.544 for a 5-word description with the next result under 0.16).

**Day 3** — Moved from a flat Python list to an actual persisted store. Expanded each movie into a full synopsis paragraph, chunked them (300 chars, ~15% overlap), embedded the chunks, and stored everything in Chroma with metadata (source title, chunk index, year, genre). Found that chunking sometimes cuts mid-sentence — one Shawshank chunk starts with "...le long-term plan culminate..." — which is a clue that came back to matter on Day 4.

**Day 4** — This is the day the project actually started talking back, and it turned into a real investigation rather than a straightforward build.

First, a bug: `generate.py` kept returning empty context for some questions, even though the data was definitely there. Turned out `ingest.py` and `generate.py` were resolving the database path differently depending on how they were run, so `generate.py` was sometimes connecting to a brand-new, silently empty database instead of the populated one. Fixed it by making every script compute the Chroma path the same way, off the script's own file location, and added a startup line that prints the resolved path and chunk count so this can't go unnoticed again.

Once that was sorted, I wired up the full pipeline (`retrieve` → `build_prompt` → call Ollama → print the answer) and ran the comparison the plan asks for: a naive prompt with no grounding instructions versus one that explicitly says to answer only from the context and admit when it doesn't know.

The naive version produced the cleanest hallucination I've seen all week. I asked "Do you know any movie about barista?" and the retrieved context had La La Land's synopsis sitting right there — "Mia, an aspiring actress working as a barista between auditions" — word for word. The model never mentioned it. Instead it invented a movie called "Barista" (2023) with a fake director attached, and cited "Like Water for Chocolate" as if it came from the provided context, which it didn't. The right answer was handed to it directly, and it ignored it completely in favor of free-associating from its own training data. That's the whole problem prompt engineering is solving, in one example.

The grounded version did much better, going 7 for 7 on a batch of questions, including two it correctly refused (Interstellar's runtime, a question about a movie that isn't in my corpus at all). That was encouraging enough that I almost called it done — until I ran one more test designed to be a clean trap. Mad Max: Fury Road's synopsis never mentions actor names, only characters (Max, Furiosa, Immortan Joe). So I asked who the main actors are, with the grounded prompt active and the *correct* Mad Max chunks genuinely retrieved. The model answered anyway — a full, accurate five-person cast list (Tom Hardy, Charlize Theron, Nicholas Hoult, Hugh Keays-Byrne, Zoë Kravitz), all factually correct, none of it in the context, despite being told explicitly not to do this. The instruction wasn't ignored because it failed to find anything — it was ignored because the model was confident enough in its own knowledge to override the rule.

That's the real takeaway from today: a grounding instruction meaningfully reduces hallucination, but it's a strong nudge, not a hard guarantee, especially for facts the model already knows cold. It held up reliably when the context was simply wrong or off-topic (the model could "tell" the topic didn't match and refused). It broke down specifically when the right document was retrieved, but the one fact being asked about wasn't in it — which is arguably the most realistic failure mode for a real production system, since it's invisible unless someone actually checks what was in the context, the way I just did.

One more thing worth flagging for Day 5: not every wrong answer is the same kind of wrong. Asking what year Get Out came out retrieved Shawshank and Grand Budapest Hotel chunks — nothing about Get Out — because the query was short and generic ("year," "come out") with nothing distinctive for the embedding to latch onto. The model then correctly refused to answer, which was the right call given what it was handed. That's a retrieval failure, not a generation failure, and they need to be measured separately — which is exactly what Day 5 is for.

## What's next

- Day 5 — evaluation (recall@k for retrieval, faithfulness for generation — measured separately, not lumped together)
- Day 6 — agent behavior
- Day 7 — polish + final write-up
- Stretch (Day 8) — pluggable LLM backend so someone cloning this repo can use their own Claude or Gemini key instead of Ollama