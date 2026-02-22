#!/usr/bin/env python3
"""
Ingest PDF rulebooks into Qdrant for RAG.

Usage:
    python ingest.py /path/to/rulebook.pdf [--collection rulebooks]
    python ingest.py /path/to/folder/       # ingests all PDFs in folder
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page from a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed(texts: list[str], config: dict) -> list[list[float]]:
    """Get embeddings from local Ollama."""
    embeddings = []
    # Ollama embed API supports batch
    resp = httpx.post(
        f"{config['ollama_url']}/api/embed",
        json={"model": config["embed_model"], "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data["embeddings"]
    return embeddings


def ensure_collection(client: QdrantClient, collection: str, vector_size: int):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection}")
    else:
        print(f"Collection exists: {collection}")


def ingest_pdf(pdf_path: str, config: dict, client: QdrantClient):
    """Ingest a single PDF into Qdrant."""
    pdf_name = Path(pdf_path).stem
    print(f"\n📖 Ingesting: {pdf_name}")

    # Extract
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Extracted {len(pages)} pages")

    # Chunk
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"], config["chunk_size"], config["chunk_overlap"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": pdf_name,
                "page": page["page"],
            })
    print(f"   Created {len(all_chunks)} chunks")

    if not all_chunks:
        print("   No text found, skipping.")
        return

    # Embed in batches of 32
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = [c["text"] for c in all_chunks[i : i + batch_size]]
        batch_embeddings = embed(batch_texts, config)
        all_embeddings.extend(batch_embeddings)
        print(f"   Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    # Ensure collection
    vector_size = len(all_embeddings[0])
    ensure_collection(client, config["qdrant_collection"], vector_size)

    # Upsert
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
            },
        )
        for chunk, emb in zip(all_chunks, all_embeddings)
    ]

    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name=config["qdrant_collection"],
            points=points[i : i + 100],
        )
    print(f"   ✅ Stored {len(points)} vectors")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant for RAG")
    parser.add_argument("path", help="PDF file or directory of PDFs")
    parser.add_argument("--collection", help="Override collection name")
    args = parser.parse_args()

    config = load_config()
    if args.collection:
        config["qdrant_collection"] = args.collection

    client = QdrantClient(url=config["qdrant_url"])

    path = Path(args.path)
    if path.is_file() and path.suffix.lower() == ".pdf":
        ingest_pdf(str(path), config, client)
    elif path.is_dir():
        pdfs = sorted(path.glob("*.pdf")) + sorted(path.glob("*.PDF"))
        if not pdfs:
            print(f"No PDFs found in {path}")
            sys.exit(1)
        for pdf in pdfs:
            ingest_pdf(str(pdf), config, client)
    else:
        print(f"Not a PDF file or directory: {path}")
        sys.exit(1)

    print("\n🎉 Ingestion complete!")


if __name__ == "__main__":
    main()
