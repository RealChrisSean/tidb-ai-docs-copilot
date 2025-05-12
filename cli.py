import os
import json
import pymysql
import boto3
import click
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Configurable Bedrock model ID for embeddings
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-g1-text-02")

# TiDB connection parameters
TIDB_CONN = {
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
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    if "embedding" in result:
        return result["embedding"]
    raise RuntimeError(f"No 'embedding' found in response: {result.keys()}")

@click.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--top_k", default=5, help="Number of results to return")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def search_cli(query, top_k, as_json):
    """Query the TiDB AI Docs Copilot from the terminal."""
    q = " ".join(query)
    # 1) Embed the query
    try:
        query_vec = get_embedding(q)
    except Exception as e:
        click.secho(f"Embedding error: {e}", fg="red")
        return

    # 2) Perform vector search
    sql = """
    SELECT source, doc_id, chunk_id, content,
           VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR)) AS score
    FROM docs_embeddings
    ORDER BY score
    LIMIT %s;
    """
    conn = pymysql.connect(**TIDB_CONN)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            vector_json = json.dumps(query_vec)
            cur.execute(sql, (vector_json, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    # 3) Handle no results
    if not rows:
        click.secho("⚠️  No results found.", fg="yellow")
        return

    # 4) Output results
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for row in rows:
            score = row["score"]
            src = row["source"]
            doc = row["doc_id"]
            snippet = row["content"].replace("\n", " ").strip()[:100]
            click.echo(f"[{score:.4f}] {src}:{doc} — {snippet}…")

if __name__ == "__main__":
    search_cli()
