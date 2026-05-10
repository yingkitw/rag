# Specification

## Purpose

Provide a Rust-first library and tools for Retrieval-Augmented Generation (RAG) that treats retrieval as three complementary signals:

1. **Vector retrieval** — dense embeddings and similarity search over chunked text.
2. **Graph retrieval** — entities and relations (for example co-occurrence in a chunk) to expand or explain context.
3. **Search-style behavior** — explicit top-k ranking, scores, pagination, and metadata filters over stored chunks.

The goal is a small, composable core that works as a library, a CLI, and an MCP server without mandating a single cloud vendor.

## Non-goals

- Owning LLM inference or chat orchestration (only retrieval and related storage).
- A hosted SaaS or multi-tenant product in this repository.
- Perfect entity resolution or world-scale knowledge graphs out of the box (the graph layer is intentionally pragmatic).

## Technical baseline

- Language: Rust, edition 2024.
- Async runtime: Tokio.
- Embeddings: OpenAI and/or local Ollama (see `OPENAI_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`). Optional **OpenAI-compatible HTTP** via [`HttpEmbeddingModel`](src/embeddings.rs).
- Vector index: pluggable `Index` trait; default exact search via `FlatIndex`; approximate **IVF** via `IvfflatIndex` (`src/index_ivf.rs`).
- Serialization: `serde` / `serde_json` (including **embeddings** on `Document` for JSON snapshots); MCP schemas via `schemars` where required for tools.

## Public surfaces

| Surface | Role |
|--------|------|
| Library (`rag` crate) | `Retriever` (vector + optional **BM25 hybrid**); `GraphRagEngine` (vector + graph + **`GraphRagSnapshot`** save/load); `Source` ingestion; `keyword` / `hybrid` / `dedup` / `rerank` helpers. |
| Binary `rag` | CLI: persistent state under `RAG_STATE_DIR` (default `.rag`): `add`, `query`, **`hybrid-query`**, `list`, `count`, **`graph-stats`**, **`graph-build`**, **`graph-hybrid-query`**. |
| Binary `rag-mcp-server` | stdio MCP server (vector + graph tools). Uses `OPENAI_API_KEY` or Ollama (`OLLAMA_URL`, **`OLLAMA_MODEL`** for embedding model name). |

## Functional requirements

### Vector path

- Chunk raw text with a pluggable `TextChunker`.
- Embed chunks with a pluggable `EmbeddingModel`.
- Store `Document` values (content, metadata, optional embedding) in a `VectorStore`.
- Query returns top-k chunks ordered by configured distance metric; optional metadata filter.
- **Lexical + vector hybrid:** `Retriever::retrieve_hybrid` merges dense search with BM25 via [`merge_hybrid`](src/hybrid.rs); optional dedup via [`dedup_similarities`](src/dedup.rs).

### Graph path

- Maintain a directed graph of `GraphNode` and `GraphEdge` in `GraphStore`.
- Extract entities from text via `EntityExtractor` (default: `SimpleEntityExtractor`: quoted strings, acronyms, proper-noun heuristics).
- Link entities that co-occur in the same chunk; relation label defaults to `co_occurs` and is **configurable** on `GraphRagEngine` (`with_co_occurrence_relation`).
- Support traversal (for example BFS), neighbor queries, community detection, and **`GraphPersisted`** / JSON **`GraphRagSnapshot`** (vectors + graph + entity↔chunk maps).

### Hybrid (vector + graph)

- `GraphRagEngine` (library) and MCP `graph_query` combine vector hits with chunks reached through entity neighborhoods to improve recall for entity-heavy queries.
- Results should remain bounded (top-k truncation) and attributable (chunk id, score when from vector path).

### Ingestion

- `Source` trait yields `ExtractedDocument` values from inputs such as PDF, codebase trees, and wiki-style URLs (see `src/ingestion.rs`).

### MCP

- Tools must remain backward-friendly where possible; graph tools are additive alongside `rag_*` vector tools.
- Server selects OpenAI embeddings when `OPENAI_API_KEY` is set, otherwise Ollama.

## Quality bar

- `cargo build` and `cargo test` succeed on supported platforms.
- No secrets in logs or error strings intended for end users.
- New retrieval behavior should be test-covered or exercised via examples when feasible.

## Documentation map

- [README.md](README.md) — quick start and feature summary.
- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout and data flow.
- [TODO.md](TODO.md) — backlog and enhancements.
