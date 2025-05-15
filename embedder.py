"""
embedder.py — take text chunks, call AWS Bedrock Titan V2, and upsert vectors into TiDB
"""
import os
import json
import pymysql
import boto3
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Configurable Bedrock embedding model ID (can override via .env).
# Default is "amazon.titan-embed-text-v1".
# If you want Titan V2 embeddings, set BEDROCK_MODEL_ID="amazon.titan-embed-text-v2:0".
# To see available models, run `aws bedrock list-models` or check the AWS console.
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")

# TiDB connection settings, pulling host/user/password from env
# TiDB connection parameters
TIDB_CONN = {
    "host": os.getenv("TIDB_HOST"),
    "port": int(os.getenv("TIDB_PORT", 4000)),
    "user": os.getenv("TIDB_USER"),
    "password": os.getenv("TIDB_PASSWORD"),
    "database": os.getenv("TIDB_DB"),
    "autocommit": True,
    "ssl": {
        "ca": os.getenv("TIDB_SSL_CA_PATH")
    },
}

# create a Bedrock client to send embedding requests
# AWS Bedrock client for embeddings
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)

# hit Bedrock Titan to get back a vector for any text
def get_embedding(text: str) -> List[float]:
    """Call Bedrock to get an embedding vector."""
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    # Parse the JSON body
    result = json.loads(response["body"].read())
    # Titan V2 returns embeddingsByType.float; fallback to top-level embedding if present
    if "embeddingsByType" in result and "float" in result["embeddingsByType"]:
        return result["embeddingsByType"]["float"]
    if "embedding" in result:
        return result["embedding"]
    raise KeyError(f"No 'embedding' found in Bedrock response: {result.keys()}")

# shove one chunk into TiDB (INSERT or UPDATE)
def upsert_embedding(conn, source: str, doc_id: str, chunk_id: int, content: str, vector: List[float]):
    """Insert or update one chunk embedding into TiDB."""
    sql = """
    INSERT INTO docs_embeddings
      (source, doc_id, chunk_id, content, embedding)
    VALUES
      (%s, %s, %s, %s, CAST(%s AS VECTOR))
    ON DUPLICATE KEY UPDATE
      content = VALUES(content),
      embedding = VALUES(embedding),
      updated_at = CURRENT_TIMESTAMP;
    """
    # turn the float list into JSON so MySQL can cast it to VECTOR
    vector_json = json.dumps(vector)
    with conn.cursor() as cur:
        cur.execute(sql, (
            source,
            doc_id,
            chunk_id,
            content,
            vector_json
        ))

# loop through all chunks, embed & upsert each to the DB
def main(chunks: List[Dict[str, Any]]):
    """Embed and upsert a list of text chunks."""
    # connect to TiDB (auto-commit on)
    conn = pymysql.connect(**TIDB_CONN)
    try:
        for idx, item in enumerate(chunks):
            emb = get_embedding(item["chunk"])
            upsert_embedding(
                conn,
                item["source"],
                str(item["id"]),
                idx,
                item["chunk"],
                emb
            )
            print(f"Upserted {item['source']}:{item['id']} chunk {idx}")
    finally:
        conn.close()

# when run directly, scrape docs & issues then embed them all
if __name__ == "__main__":
    from scraper import fetch_docs, fetch_github_issues
    all_chunks = fetch_docs() + fetch_github_issues()
    main(all_chunks)