from typing import List
import os
import json
import pymysql
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from scraper import fetch_docs, fetch_github_issues
from embedder import main as run_embedder
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

import os
# Compute the absolute path to the project root
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = BASE_DIR

# Configurable Bedrock model ID for embeddings
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-g1-text-02")

# Bedrock client for embeddings
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)

# ---------- Local LLM Setup (FLAN-T5-small) ----------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
rag_pipe  = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False
)

# ---------- Config & Clients ----------
# TiDB connection
TIDB = {
    "host":     os.getenv("TIDB_HOST"),
    "port":     int(os.getenv("TIDB_PORT", 4000)),
    "user":     os.getenv("TIDB_USER"),
    "password": os.getenv("TIDB_PASSWORD"),
    "database": os.getenv("TIDB_DB"),
    "autocommit": True,
    "ssl": {
        "ca": os.getenv("TIDB_SSL_CA_PATH")
    },
}

def get_embedding(text: str) -> List[float]:
    resp = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(resp["body"].read())
    # Bedrock embedding responses use "embedding" as the key
    if "embedding" in result:
        return result["embedding"]
    raise HTTPException(status_code=500, detail=f"No 'embedding' field in response: {result.keys()}")

# ---------- FastAPI App & Models ----------
app = FastAPI(title="TiDB AI Docs Copilot")

# Serve static assets under /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve the main HTML page at the root path
@app.get("/", include_in_schema=False)
def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ----------- Scheduled and Manual Refresh -----------
# (Assume the scheduler and fetch_docs, fetch_github_issues, run_embedder are defined elsewhere)
# If not, you would insert this after the scheduler.start() call.

# Manual refresh endpoint
@app.post("/refresh", include_in_schema=False)
def manual_refresh():
    """
    Manually trigger the embedding refresh.
    """
    try:
        chunks = fetch_docs() + fetch_github_issues()
        run_embedder(chunks)
        return JSONResponse({"status": "ok", "refreshed_chunks": len(chunks)})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


class SearchResult(BaseModel):
    source: str
    doc_id: str
    chunk_id: int
    content: str
    score: float

# Request model for /answer endpoint
class AnswerRequest(BaseModel):
    question: str
    top_k: int = 5

# ---------- /search Endpoint ----------
@app.get("/search", response_model=List[SearchResult])
def search(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)) -> List[SearchResult]:
    # 1) Embed the query
    try:
        query_vec = get_embedding(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Run vector search in TiDB
    sql = """
    SELECT
      source,
      doc_id,
      chunk_id,
      content,
      VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR)) AS score
    FROM docs_embeddings
    ORDER BY score
    LIMIT %s;
    """
    conn = pymysql.connect(**TIDB)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            vector_json = json.dumps(query_vec)
            cur.execute(sql, (vector_json, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    return rows

# ---------- /answer Endpoint ----------
@app.post("/answer")
def answer(req: AnswerRequest):
    # 1) Embed the question
    try:
        query_vec = get_embedding(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Fetch top_k contexts
    sql = """
      SELECT content 
      FROM docs_embeddings
      ORDER BY VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR))
      LIMIT %s;
    """
    conn = pymysql.connect(**TIDB)
    try:
        with conn.cursor() as cur:
            vector_json = json.dumps(query_vec)
            cur.execute(sql, (vector_json, req.top_k))
            contexts = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    # Prioritize contexts containing any question keywords
    keywords = req.question.lower().split()
    filtered = [c for c in contexts if any(kw in c.lower() for kw in keywords)]
    contexts = filtered if filtered else contexts

    # Only keep up to 3 contexts
    contexts = contexts[:3]

    # Build the RAG prompt with a stronger instruction
    prompt = (
        "You are a knowledgeable TiDB documentation assistant. Read the following official TiDB docs excerpts and write a concise, informative answer in 2-4 sentences.\n\n"
        "Excerpts:\n" +
        "\n\n".join(f"- {c}" for c in contexts) +
        f"\n\nQuestion: {req.question}\nAnswer:"
    )

    # Generate the final answer via local FLAN-T5-small pipeline
    try:
        output = rag_pipe(prompt)[0]["generated_text"]
    except Exception as e:
        tb = traceback.format_exc()
        print("🛑 Local LLM inference error:", tb)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 5) Return the cleaned-up answer plus source contexts
    return {
        "answer": output,
        "sources": contexts
    }   