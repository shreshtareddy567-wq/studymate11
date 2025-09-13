# backend/main.py
import os
import io
import math
import json
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# PDF parsing
from PyPDF2 import PdfReader

# embeddings + models
try:
    from sentence_transformers import SentenceTransformer
    use_sentence_transformers = True
except Exception:
    use_sentence_transformers = False

import numpy as np
import requests

# ---- CONFIG ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # adjust if you prefer
OPENAI_CHAT_MODEL = "gpt-4o"  # or gpt-4o-mini / gpt-4o-mini-instruct — tune as you like

# If sentence_transformers is present, we'll use it locally for embeddings (faster/no cost).
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, reasonably good

CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks
TOP_K = 5  # how many chunks to fetch for context

# ---- In-memory vector store structure ----
# { doc_id: { "chunks": [ {text, page, chunk_id, embedding (np.array)} ], "meta": {...} } }
VECTOR_STORE: Dict[str, Any] = {}

# ---- App ----
app = FastAPI(title="StudyMate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Utilities ----
def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[Dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": i + 1, "text": text})
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    chunk_id = 0
    while start < length:
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append((chunk_id, start, piece))
            chunk_id += 1
        start = max(start + chunk_size - overlap, end)
    return chunks

# embedding helpers
if use_sentence_transformers:
    EMB_MODEL = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
    def embed_texts(texts: List[str]) -> np.ndarray:
        # returns numpy array shape (n, dim)
        return EMB_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
else:
    def embed_texts(texts: List[str]) -> np.ndarray:
        # Use OpenAI embeddings endpoint
        if not OPENAI_API_KEY:
            raise RuntimeError("No local embedding model installed and OPENAI_API_KEY not set.")
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        embeddings = []
        # call in batches if needed
        for txt in texts:
            payload = {"input": txt, "model": OPENAI_EMBEDDING_MODEL}
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            embeddings.append(j["data"][0]["embedding"])
        return np.array(embeddings, dtype=float)

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a shape (d,), b shape (n, d) or vice versa
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(b, axis=1)
    if a_norm == 0 or np.any(b_norms == 0):
        return np.zeros(b.shape[0])
    sims = np.dot(b, a) / (b_norms * a_norm)
    return sims

# ---- Pydantic models ----
class UploadResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    message: str

class AskRequest(BaseModel):
    doc_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# ---- Endpoints ----
@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pages = extract_text_from_pdf_bytes(contents)
    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    doc_id = file.filename + "_" + str(len(VECTOR_STORE) + 1)
    chunks_meta = []
    all_texts = []
    for p in pages:
        text = p["text"] or ""
        if not text.strip():
            continue
        page_chunks = chunk_text(text)
        for cid, start, piece in page_chunks:
            chunks_meta.append({"chunk_id": f"{cid}", "page": p["page_number"], "text": piece})
            all_texts.append(piece)
    if not all_texts:
        raise HTTPException(status_code=400, detail="PDF contained no extractable text.")
    # compute embeddings
    try:
        embeddings = embed_texts(all_texts)  # shape (n, dim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failure: {e}")
    # store
    stored_chunks = []
    for i, m in enumerate(chunks_meta):
        stored_chunks.append({
            "chunk_id": m["chunk_id"],
            "page": m["page"],
            "text": m["text"],
            "embedding": embeddings[i].tolist()  # store as list for JSON safety
        })
    VECTOR_STORE[doc_id] = {"meta": {"filename": file.filename}, "chunks": stored_chunks}
    return UploadResponse(doc_id=doc_id, chunks_indexed=len(stored_chunks), message="Indexed successfully")

@app.post("/ask", response_model=AnswerResponse)
async def ask(req: AskRequest):
    if req.doc_id not in VECTOR_STORE:
        raise HTTPException(status_code=404, detail="doc_id not found. Please upload first.")
    # embed the question
    try:
        q_emb = embed_texts([req.question])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failure: {e}")
    # compute similarities against stored chunks
    chunks = VECTOR_STORE[req.doc_id]["chunks"]
    corpus_embeddings = np.array([np.array(c["embedding"], dtype=float) for c in chunks])
    sims = cosine_sim(q_emb, corpus_embeddings)  # shape (n,)
    top_idx = sims.argsort()[-TOP_K:][::-1]
    top_chunks = [{"score": float(sims[i]), "page": chunks[i]["page"], "text": chunks[i]["text"]} for i in top_idx]
    # build the context
    context_text = "\n\n---\n\n".join([f"(page {c['page']}) {c['text']}" for c in top_chunks])
    # Query LLM (OpenAI ChatCompletion). We'll use the simple Chat Completions HTTP API
    if not OPENAI_API_KEY:
        # If no API key, return a deterministic fallback combining top chunks
        combined_answer = "I don't have an LLM API key configured. Here are the most relevant passages:\n\n" + context_text
        return AnswerResponse(answer=combined_answer, sources=top_chunks)
    # Compose prompt
    system_prompt = (
        "You are StudyMate, an AI assistant for students. Use the provided context extracted from a PDF to answer the user's question. "
        "Cite page numbers where appropriate. If the answer is not contained in the context, say so and try to answer generally."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {req.question}\n\nAnswer concisely, then list which context pages you used."
    chat_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "n": 1,
    }
    r = requests.post(chat_url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {r.text}")
    j = r.json()
    answer_text = j["choices"][0]["message"]["content"].strip()
    return AnswerResponse(answer=answer_text, sources=top_chunks)

@app.get("/list_docs")
async def list_docs():
    return [{"doc_id": k, "filename": v["meta"]["filename"], "chunks": len(v["chunks"])} for k, v in VECTOR_STORE.items()]

# ---- Run ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
// frontend/src/App.js
import React, { useState, useEffect } from "react";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [docId, setDocId] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [docs, setDocs] = useState([]);

  useEffect(() => {
    fetchDocs();
  }, []);

  async function fetchDocs() {
    try {
      const res = await fetch(`${API_BASE}/list_docs`);
      const j = await res.json();
      setDocs(j);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!pdfFile) return alert("Select a PDF first.");
    setUploadStatus("Uploading...");
    const fd = new FormData();
    fd.append("file", pdfFile, pdfFile.name);
    try {
      const res = await fetch(`${API_BASE}/upload_pdf`, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const j = await res.json();
      setDocId(j.doc_id);
      setUploadStatus(`Indexed: ${j.chunks_indexed} chunks. doc_id=${j.doc_id}`);
      fetchDocs();
    } catch (err) {
      setUploadStatus("Upload failed: " + err.message);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    if (!docId) return alert("Select/upload a document first (doc_id).");
    setAnswer("Thinking...");
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ doc_id: docId, question }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Ask failed");
      }
      const j = await res.json();
      setAnswer(j.answer);
      setSources(j.sources || []);
    } catch (err) {
      setAnswer("Error: " + err.message);
    }
  }

  return (
    <div style={{maxWidth: 900, margin: "20px auto", fontFamily: "Arial, sans-serif"}}>
      <h1>StudyMate — PDF Q&A</h1>

      <section style={{padding: 12, border: "1px solid #ddd", borderRadius: 8, marginBottom: 18}}>
        <h2>1) Upload PDF</h2>
        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setPdfFile(e.target.files[0])}
          />
          <button type="submit" style={{marginLeft: 8}}>Upload & Index</button>
        </form>
        <div style={{marginTop: 8, color: "#333"}}>{uploadStatus}</div>
        <div style={{marginTop: 8}}>
          <strong>Available docs:</strong>
          <ul>
            {docs.map(d => (
              <li key={d.doc_id}>
                <button onClick={() => setDocId(d.doc_id)}>Use</button>
                {" "} {d.filename} — chunks: {d.chunks} — id: <code>{d.doc_id}</code>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section style={{padding: 12, border: "1px solid #ddd", borderRadius: 8}}>
        <h2>2) Ask a question</h2>
        <div style={{marginBottom: 8}}>
          <label>Using doc_id: </label>
          <input value={docId} onChange={(e) => setDocId(e.target.value)} style={{width: 400}} />
        </div>
        <form onSubmit={handleAsk}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask something about the uploaded PDF..."
            rows={4}
            style={{width: "100%"}}
          />
          <button type="submit" style={{marginTop: 8}}>Ask</button>
        </form>
        <div style={{marginTop: 12}}>
          <h3>Answer</h3>
          <div style={{whiteSpace: "pre-wrap", background: "#fafafa", padding: 12, borderRadius: 6}}>
            {answer}
          </div>
          <h4>Sources (top matches)</h4>
          <ol>
            {sources.map((s, i) => (
              <li key={i}>
                <strong>Page {s.page}</strong> (score {s.score.toFixed(3)}) — <div style={{whiteSpace: "pre-wrap"}}>{s.text.slice(0,400)}{s.text.length>400?"...":""}</div>
              </li>
            ))}
          </ol>
        </div>
      </section>

    </div>
  );
}

export default App;
