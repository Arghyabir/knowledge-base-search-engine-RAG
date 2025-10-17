import os
import json
import uuid
from utils.pdf_reader import read_pdf
from sentence_transformers import SentenceTransformer
from embeddings_store import EmbeddingsStore

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    start = 0
    chunks = []
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def ingest_files(file_paths, index_dir="vector_store"):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    store = EmbeddingsStore(index_dir=index_dir, model=model)
    meta_items = []
    for path in file_paths:
        ext = path.split(".")[-1].lower()
        if ext == "pdf":
            text = read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            metadata = {"source": os.path.basename(path), "chunk_id": i, "doc_id": doc_id}
            emb = model.encode(c)
            store.add(emb, metadata, text=c)
            meta_items.append((doc_id, metadata))
    store.save()
    return meta_items

if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python ingest_docs.py doc1.pdf doc2.txt ...")
        sys.exit(1)
    print("Ingesting:", paths)
    ingest_files(paths)
    print("Done")
