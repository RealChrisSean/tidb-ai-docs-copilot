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
