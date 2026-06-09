# TODO

## In progress

- [x] Incremental `graph-build`: CLI loads existing `graph_rag.json` and merges new documents instead of overwriting.
- [x] `GraphRagEngine::add_document_with_metadata` so `graph-build --source` is actually stored.
- [x] Register all `examples/` in `Cargo.toml` for explicit discoverability.
- [x] Align `SPEC.md` binary name (`rag-mcp` not `rag-mcp-server`).

## Shipped in this repo

- Hybrid BM25 + vector in `Retriever`, IVF index (`IvfflatIndex`), JSON auto-flush store (`JsonPersistentVectorStore`), graph snapshot (`GraphRagSnapshot` / save + load), configurable co-occurrence relation, `HttpEmbeddingModel`, CLI state dir (`RAG_STATE_DIR`) with `hybrid-query`, `graph-stats`, `graph-build`, `graph-hybrid-query`.
- Hygiene: `documentation` key fixed in `Cargo.toml`. Embeddings included in JSON persistence for `Document`.

## Near-term backlog

- [x] `rag add` should accept a directory and ingest all `.txt` / `.md` files recursively.
- [x] `rag add` should accept multiple `--file` arguments (batch ingestion).
- [x] CLI `--metric` flag to choose distance metric at runtime (cosine, euclidean, dot, manhattan).
- [x] CLI `--chunker` flag to switch chunking strategy at runtime (fixed, paragraph, sentence).
- [x] `cargo audit` in CI (add `.github/workflows/ci.yml`).
- [x] Unit tests for `main.rs` CLI argument parsing and routing.

## Optional / research

- [ ] PostgreSQL / Qdrant / remote vector backends implementing `VectorStore`.
- [x] Full HNSW (`hnsw_rs` crate) implementing `Index`; IVF is a first ANN step.
- [x] LLM-assisted `EntityExtractor` behind a feature flag (`llm-extractor`).

## Maintenance

- [ ] Keep [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [README.md](README.md) aligned when behavior changes.
