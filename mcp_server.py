#!/usr/bin/env python3
"""
MCP server that provides RAG search over rulebooks stored in Qdrant.

Exposes a 'search_rules' tool that queries Qdrant using semantic search
and returns relevant rulebook chunks with source citations.

Run: python mcp_server.py                (stdio transport)
     python mcp_server.py --http         (streamable HTTP on port 8100)
"""

import argparse
import json
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from qdrant_client import QdrantClient

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CONFIG = load_config()
qdrant = QdrantClient(url=CONFIG["qdrant_url"])
mcp = FastMCP("rag-rulebooks", host="0.0.0.0", port=8100)

# --- Prometheus metrics ---
SEARCH_REQUESTS = Counter(
    "rag_search_requests_total",
    "Total search_rules calls",
    ["source_filter"],
)
SEARCH_LATENCY = Histogram(
    "rag_search_latency_seconds",
    "search_rules latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
SEARCH_RESULTS = Histogram(
    "rag_search_results_count",
    "Number of results returned per search",
    buckets=[0, 1, 2, 3, 5, 10],
)
EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds",
    "Ollama embedding call latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)
SERVER_INFO = Info("rag_server", "RAG MCP server info")
SERVER_INFO.info({
    "collection": CONFIG["qdrant_collection"],
    "embed_model": CONFIG["embed_model"],
})


def get_embedding(text: str) -> list[float]:
    """Get embedding from local Ollama."""
    start = time.monotonic()
    resp = httpx.post(
        f"{CONFIG['ollama_url']}/api/embed",
        json={"model": CONFIG["embed_model"], "input": [text]},
        timeout=30.0,
    )
    resp.raise_for_status()
    EMBEDDING_LATENCY.observe(time.monotonic() - start)
    return resp.json()["embeddings"][0]


@mcp.tool()
def search_rules(query: str, top_k: int = 5, source_filter: str = "") -> str:
    """Search Bolt Action and Konflikt '47 rulebooks for relevant rules,
    stats, army lists, or game mechanics. Returns the most relevant
    passages with page numbers and source book.

    Args:
        query: Natural language question about rules, units, mechanics, etc.
        top_k: Number of results to return (default 5, max 10).
        source_filter: Optional: filter to a specific rulebook by name.
    """
    top_k = min(top_k, 10)
    SEARCH_REQUESTS.labels(source_filter=source_filter or "all").inc()
    search_start = time.monotonic()

    # Embed the query
    query_vector = get_embedding(query)

    # Build search filter
    search_filter = None
    if source_filter:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    # Search
    from qdrant_client.models import models

    results = qdrant.query_points(
        collection_name=CONFIG["qdrant_collection"],
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
    )

    SEARCH_LATENCY.observe(time.monotonic() - search_start)
    SEARCH_RESULTS.observe(len(results.points))

    if not results.points:
        return "No relevant rules found for that query."

    # Format results
    output_parts = [f"## Results for: {query}\n"]
    for i, hit in enumerate(results.points, 1):
        payload = hit.payload
        score = hit.score
        output_parts.append(
            f"### {i}. {payload['source']} — Page {payload['page']} (score: {score:.3f})\n"
            f"{payload['text']}\n"
        )

    return "\n".join(output_parts)


@mcp.tool()
def list_sources() -> str:
    """List all ingested rulebooks/sources available for search."""
    try:
        sources = set()
        offset = None
        while True:
            records, offset = qdrant.scroll(
                collection_name=CONFIG["qdrant_collection"],
                limit=100,
                offset=offset,
                with_payload=["source"],
            )
            for r in records:
                sources.add(r.payload.get("source", "unknown"))
            if offset is None:
                break

        if not sources:
            return "No rulebooks have been ingested yet."

        source_list = "\n".join(f"- {s}" for s in sorted(sources))
        return f"## Ingested Rulebooks\n{source_list}"
    except Exception as e:
        return f"Error listing sources: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run as Streamable HTTP server on port 8100")
    parser.add_argument("--metrics-port", type=int, default=9100, help="Prometheus metrics port (default 9100)")
    args = parser.parse_args()

    if args.http:
        # Start Prometheus metrics server on a separate port
        from prometheus_client import start_http_server
        start_http_server(args.metrics_port, addr="0.0.0.0")
        print(f"Prometheus metrics at http://0.0.0.0:{args.metrics_port}/metrics")

        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
