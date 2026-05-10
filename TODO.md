# TODO

## Near term

- [x] Add runnable `graph_rag_example`, offline `graph_store_basic`, and fixture-driven `ingest_fixture_rag` under `examples/`.
- [x] Add deterministic `GraphRagEngine` coverage in `tests/graph_rag_engine_test.rs` and graph-store pattern test in `example_behavior_test.rs`.
- [ ] Extend integration tests for empty-graph / no-entity edge cases beyond current suite.
- [ ] Document `OLLAMA_MODEL` for `rag-mcp` in README environment section.

## Retrieval and search

- [ ] Library-level hybrid or BM25-style keyword retrieval to complement pure vector search (MCP already combines vector + graph; align library API).
- [ ] Optional cross-encoder or reranking stage behind a trait (pluggable, default noop).
- [ ] Deduplication of near-duplicate chunks at ingest or query time.

## Storage and scale

- [ ] Persistent `VectorStore` backends (for example PostgreSQL + pgvector, Qdrant, or file-based).
- [ ] Approximate nearest neighbor `Index` implementation (HNSW or IVF) for large corpora; keep `FlatIndex` as default for small data.
- [ ] Graph persistence and load/save aligned with vector store lifecycle.

## Models and extraction

- [ ] Additional `EmbeddingModel` implementations (vendor-agnostic HTTP or popular local runners) behind the same trait.
- [ ] Optional LLM- or NER-based `EntityExtractor` for higher-quality graphs; keep `SimpleEntityExtractor` as default.
- [ ] Configurable relation types beyond `co_occurs` where extraction supports it.

## Product and DX

- [ ] CLI commands for graph stats and optional `graph_query` parity with MCP.
- [ ] Publish or tighten `docs.rs` descriptions for new public types.
- [ ] Versioning and migration notes for stored graphs / indices when persistence lands.

## Hygiene

- [ ] Periodically run `cargo audit` / dependency updates per project policy.
- [ ] Keep SPEC / ARCHITECTURE / README in sync after behavioral changes to tools or public API.
