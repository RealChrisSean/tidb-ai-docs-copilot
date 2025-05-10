import os
import json
import pymysql
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
from typing import List

load_dotenv()

# ---------- Config & Clients ----------
# TiDB connection
TIDB = {
    "host":     os.getenv("TIDB_HOST"),
    "port":     int(os.getenv("TIDB_PORT", 4000)),
    "user":     os.getenv("TIDB_USER"),
    "password": os.getenv("TIDB_PASSWORD"),
    "database": os.getenv("TIDB_DB"),
    "autocommit": True,
}

# Bedrock client for embeddings
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)

def get_embedding(text: str) -> List[float]:
    resp = bedrock.invoke_model(
        modelId="amazon.titan-text-embedding",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    return json.loads(resp["body"].read())["embeddings"]

# ---------- FastAPI App & Models ----------
app = FastAPI(title="TiDB AI Docs Copilot")

class SearchResult(BaseModel):
    source: str
    doc_id: str
    chunk_id: int
    content: str
    score: float

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
      VEC_COSINE_DISTANCE(embedding, %s) AS score
    FROM docs_embeddings
    ORDER BY score
    LIMIT %s;
    """
    conn = pymysql.connect(**TIDB)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, (query_vec, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    return rows