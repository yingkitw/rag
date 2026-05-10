# TODO

## Shipped in this repo

- Near-term tests, hybrid BM25 + vector in `Retriever`, IVF index (`IvfflatIndex`), JSON auto-flush store (`JsonPersistentVectorStore`), graph snapshot (`GraphRagSnapshot` / save + load), configurable co-occurrence relation, `HttpEmbeddingModel`, CLI state dir (`RAG_STATE_DIR`) with `hybrid-query`, `graph-stats`, `graph-build`, `graph-hybrid-query`.
- Hygiene: `documentation` key fixed in `Cargo.toml` (replaces invalid `package.docs`). Embeddings are now included in JSON persistence for `Document`.

## Optional follow-ups

- [ ] Install `cargo-audit` and add it to CI: `cargo install cargo-audit && cargo audit`.
- [ ] PostgreSQL / Qdrant / remote vector backends implementing `VectorStore`.
- [ ] Full HNSW (e.g. external crate) implementing `Index`; IVF is a first ANN step.
- [ ] LLM-assisted `EntityExtractor` behind a feature flag.
- [ ] CLI: merge multiple files into one snapshot; incremental `graph-build`.

## Maintenance

- [ ] Keep [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [README.md](README.md) aligned when behavior changes.
