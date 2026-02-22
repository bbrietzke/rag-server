# RAG Server for Wargaming Rulebooks

Semantic search over Bolt Action / Konflikt '47 rulebooks via Qdrant + Ollama embeddings, exposed as an MCP server for OpenClaw.

## Architecture

```
PDFs → ingest.py → Ollama (nomic-embed-text) → Qdrant
                                                  ↑
OpenClaw agent → mcporter → mcp_server.py → Qdrant → relevant chunks
```

Everything runs locally on the same host.

## Setup

### 1. Install Qdrant

```bash
# Copy systemd unit
sudo cp systemd/qdrant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qdrant

# Verify
curl http://localhost:6333/healthz
```

### 2. Install Ollama + embedding model

```bash
# If not already installed
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
```

### 3. Install RAG server

```bash
# Copy to /opt
sudo cp -r . /opt/rag-server
cd /opt/rag-server

# Create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install systemd unit
sudo cp systemd/rag-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-mcp
```

### 4. Ingest rulebooks

```bash
cd /opt/rag-server
source venv/bin/activate

# Single PDF
python ingest.py /path/to/bolt-action-rulebook.pdf

# Whole folder of PDFs
python ingest.py /path/to/rulebooks/
```

### 5. Connect to OpenClaw via mcporter

mcporter looks for config in two places (in order):

1. **Project config:** `<openclaw-workspace>/config/mcporter.json`
2. **System config:** `~/.mcporter/mcporter.json`

For the wargaming agent, create the project config:

```bash
mkdir -p /home/<user>/.openclaw/workspace/config
cat > /home/<user>/.openclaw/workspace/config/mcporter.json << 'EOF'
{
  "servers": {
    "rulebooks": {
      "baseUrl": "http://localhost:8100/sse"
    }
  }
}
EOF
```

Or use the CLI:

```bash
mcporter config add rulebooks --url http://localhost:8100/sse --transport sse
```

Verify it works:

```bash
mcporter list              # should show "rulebooks" server
mcporter list rulebooks    # should show search_rules + list_sources tools
```

Then the agent gets two tools:
- **search_rules** — semantic search with query, optional source filter
- **list_sources** — list all ingested books

## Config

Edit `config.json` to adjust:
- `chunk_size` / `chunk_overlap` — text chunking params
- `top_k` — default number of results
- `embed_model` — Ollama model for embeddings
- Qdrant/Ollama URLs (default: localhost)

## Adding more books later

Just run `ingest.py` again with new PDFs. Vectors append to the same collection.
