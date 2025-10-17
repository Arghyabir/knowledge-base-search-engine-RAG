# Knowledge-base Search Engine (RAG)

## Overview
Simple RAG system:
- Ingest PDFs / text
- Create embeddings (sentence-transformers)
- Store in FAISS
- Query: retrieve top-k chunks and synthesize answer via OpenAI

## Setup
1. Create virtualenv and install deps:
   pip install -r requirements.txt

2. Set environment variables:
   export OPENAI_API_KEY="sk-..."
   (or create .env with OPENAI_API_KEY=...)

3. Run backend:
   uvicorn app:app --reload --port 8000

4. UI:
   open static/index.html and point API to http://localhost:8000/query

## Endpoints
- POST /ingest  (multipart: files[])
- POST /query   (json: {"query": "...", "k": 5})

## Notes
- Replace OpenAI with any LLM by editing generate_answer.py
- Keep a copy of FAISS index and metadata file (saved automatically)
