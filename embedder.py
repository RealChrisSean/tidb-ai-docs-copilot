import os
import json
import pymysql
import boto3
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

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

# AWS Bedrock client for embeddings
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)

def get_embedding(text: str) -> List[float]:
    """Call Bedrock to get an embedding vector."""
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-embedding",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    return result["embeddings"]

def upsert_embedding(conn, source: str, doc_id: str, chunk_id: int, content: str, vector: List[float]):
    """Insert or update one chunk embedding into TiDB."""
    sql = """
    INSERT INTO docs_embeddings
      (source, doc_id, chunk_id, content, embedding)
    VALUES
      (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      content = VALUES(content),
      embedding = VALUES(embedding),
      updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            source,
            doc_id,
            chunk_id,
            content,
            vector
        ))

def main(chunks: List[Dict[str, Any]]):
    """Embed and upsert a list of text chunks."""
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

if __name__ == "__main__":
    from scraper import fetch_docs, fetch_github_issues
    all_chunks = fetch_docs() + fetch_github_issues()
    main(all_chunks)