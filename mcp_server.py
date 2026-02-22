#!/usr/bin/env python3
"""
MCP server that provides RAG search over rulebooks stored in Qdrant.

Exposes a 'search_rules' tool that queries Qdrant using semantic search
and returns relevant rulebook chunks with source citations.

Run: python mcp_server.py          (stdio transport, for mcporter)
     python mcp_server.py --http   (HTTP transport on port 8100)
"""

import argparse
import json
from pathlib import Path

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from qdrant_client import QdrantClient

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CONFIG = load_config()
qdrant = QdrantClient(url=CONFIG["qdrant_url"])
server = Server("rag-rulebooks")


def get_embedding(text: str) -> list[float]:
    """Get embedding from local Ollama."""
    resp = httpx.post(
        f"{CONFIG['ollama_url']}/api/embed",
        json={"model": CONFIG["embed_model"], "input": [text]},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_rules",
            description=(
                "Search Bolt Action and Konflikt '47 rulebooks for relevant rules, "
                "stats, army lists, or game mechanics. Returns the most relevant "
                "passages with page numbers and source book."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question about rules, units, mechanics, etc.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                        "default": 5,
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Optional: filter to a specific rulebook by name.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_sources",
            description="List all ingested rulebooks/sources available for search.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_rules":
        return await search_rules(arguments)
    elif name == "list_sources":
        return await list_sources()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_rules(args: dict) -> list[TextContent]:
    query = args["query"]
    top_k = min(args.get("top_k", CONFIG["top_k"]), 10)
    source_filter = args.get("source_filter")

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
    results = qdrant.search(
        collection_name=CONFIG["qdrant_collection"],
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k,
    )

    if not results:
        return [TextContent(type="text", text="No relevant rules found for that query.")]

    # Format results
    output_parts = [f"## Results for: {query}\n"]
    for i, hit in enumerate(results, 1):
        payload = hit.payload
        score = hit.score
        output_parts.append(
            f"### {i}. {payload['source']} — Page {payload['page']} (score: {score:.3f})\n"
            f"{payload['text']}\n"
        )

    return [TextContent(type="text", text="\n".join(output_parts))]


async def list_sources() -> list[TextContent]:
    """List unique sources in the collection."""
    try:
        # Scroll through to get unique sources
        records, _ = qdrant.scroll(
            collection_name=CONFIG["qdrant_collection"],
            limit=1,
            with_payload=["source"],
        )
        # Get all unique sources via scroll
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
            return [TextContent(type="text", text="No rulebooks have been ingested yet.")]

        source_list = "\n".join(f"- {s}" for s in sorted(sources))
        return [TextContent(type="text", text=f"## Ingested Rulebooks\n{source_list}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing sources: {e}")]


async def main_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run as HTTP server on port 8100")
    args = parser.parse_args()

    if args.http:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1])

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        ])
        uvicorn.run(app, host="127.0.0.1", port=8100)
    else:
        asyncio.run(main_stdio())
