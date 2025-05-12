# TiDB AI Docs Copilot

A “self-updating” AI assistant that scrapes your TiDB docs and GitHub issues, embeds them with AWS Bedrock, and stores vectors in TiDB for semantic search.

## Command-Line Interface

You can query the Docs Copilot directly from your terminal:

```bash
# Pretty print results
python cli.py "your query here" --top_k 5

# Raw JSON output
python cli.py "your query here" --top_k 5 --json
```

## Web Interface

You can also query the Docs Copilot via a browser UI:

1. Start the server:
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
2. Open your browser at [http://localhost:8000/](http://localhost:8000/).
3. Enter your question in the search box and hit **Search**.
4. The AI-generated answer appears at the top, with source excerpts below.

## Answer Endpoint

If you prefer raw JSON output from your scripts, use the `/answer` endpoint:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Your question here","top_k":5}'
```

The response will be:

```json
{
  "answer": "A concise answer generated from TiDB docs",
  "sources": [
    "Excerpt from doc 1...",
    "Excerpt from doc 2..."
  ]
}
```

## Manual Refresh

To manually refresh the embeddings (scrape docs & GitHub issues, re-embed, and upsert), call:

```bash
curl -X POST http://localhost:8000/refresh
```

You should receive a JSON response indicating how many chunks were refreshed.
