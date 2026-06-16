# bRAG About It

A small command-line tool that answers questions over a set of documents I choose — built from scratch over a week to show Retrieval Augmented Generation end-to-end.

## Status: Day 1 of 7 — project setup + first LLM call

## Stack

| Piece | Choice | Why |
|---|---|---|
| LLM | Ollama (`llama3.2`) | Local, free, no API key needed |
| Embeddings | `sentence-transformers` | Local, free, no rate limits |
| Vector store | Chroma | Pure Python, persists to disk |

## Setup

1. Install [Ollama](https://ollama.com) and pull a model: ollama pull llama3.2 
2. Create a virtual environment and install dependencies: pip install -r requirements.txt

## Progress log

- **Day 1** — repo set up, Ollama running locally, first script-generated LLM response.

## What's next

- Day 2 — embeddings + semantic search
- Day 3 — chunking + Chroma vector store
- Day 4 — retrieval + generation + prompt engineering
- Day 5 — evaluation
- Day 6 — agent behavior
- Day 7 — polish + final write-up