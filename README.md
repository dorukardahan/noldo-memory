# 🧠 Asuman Memory

Production-ready conversational memory for [Asuman](https://github.com/asuman-project) — an AI assistant running on OpenClaw.

Turkish+English hybrid search, OpenRouter embeddings, knowledge graph, temporal awareness.

## Architecture

```
OpenClaw Gateway (Node.js, WhatsApp)
    │
    │ HTTP localhost:8787
    ▼
Asuman Memory (Python)
├── OpenRouter embeddings (qwen/qwen3-embedding-8b)
├── sqlite-vec + FTS5 (hybrid search)
├── Turkish NLP (zeyrek + dateparser)
├── Trigger patterns (TR+EN)
├── Knowledge graph (SQLite)
├── RRF fusion (semantic + BM25 + recency)
└── Confidence scoring
```

## Based On

Enhanced fork inspired by [Mahmory](https://github.com/cryptosquanch/whatsapp-memory) (v6.0) — rebuilt from scratch with:
- 🪶 **~20MB** dependencies (vs ~4GB original)
- 🇹🇷 **Turkish NLP** — zeyrek morphology, dateparser temporal, Turkish triggers
- 🔗 **OpenRouter** embeddings — qwen3-embedding-8b (MTEB Multilingual #1)
- 💾 **sqlite-vec** — single file, hybrid search, trivial backup
- ⚡ **FastAPI** — HTTP bridge to OpenClaw

## Status

🚧 Under development — see [BUILD-PLAN.md](BUILD-PLAN.md) for roadmap.

## License

Private repository.
