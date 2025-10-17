import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ingest_docs import ingest_files
from retriever import Retriever
from generate_answer import synthesize_answer
import shutil
import uuid

app = FastAPI(title="RAG Knowledge-base API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever = Retriever(index_dir="vector_store")

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        filename = f"{uuid.uuid4()}_{f.filename}"
        out_path = os.path.join(UPLOAD_DIR, filename)
        with open(out_path, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_paths.append(out_path)
    # ingest synchronously (for simplicity)
    meta = ingest_files(saved_paths, index_dir="vector_store")
    return {"status":"ok","ingested_files": [p.split("_",1)[1] for p in saved_paths], "meta_count": len(meta)}

@app.post("/query")
async def query(payload: dict):
    q = payload.get("query")
    k = int(payload.get("k",5))
    if not q:
        return JSONResponse({"error":"query missing"}, status_code=400)
    # retrieve
    retrieved = retriever.retrieve(q, k=k)
    # synthesize
    answer = synthesize_answer(q, retrieved)
    return {"query": q, "answer": answer, "retrieved": retrieved}

@app.get("/")
def home():
    html = open("static/index.html").read()
    return HTMLResponse(content=html)
