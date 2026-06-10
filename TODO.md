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
- [x] Reciprocal Rank Fusion (RRF) as an alternative to score-based `merge_hybrid` for vector + BM25 merging.
- [x] Configurable BM25 parameters (`k1`, `b`) instead of hardcoded constants.
- [x] Result diversification / `limit.per` metadata attribute (e.g., cap chunks per source file).
- [x] External reranker implementations (Cohere, Voyage, MixedBread) behind `SimilarityReranker` trait.
- [x] Exact kNN search over filtered subsets (guaranteed precision when filter narrows the space).

## Optional / research

- [ ] PostgreSQL / Qdrant / remote vector backends implementing `VectorStore`.
- [x] Full HNSW (`hnsw_rs` crate) implementing `Index`; IVF is a first ANN step.
- [x] LLM-assisted `EntityExtractor` behind a feature flag (`llm-extractor`).
- [x] Sparse vector index and search (e.g., SPLADE-style lexical semantic signals).
- [x] Query rewriting / multi-query generation with LLM (generate variants, run in parallel, fuse).
- [x] Phrase matching and prefix queries in BM25 (beyond token-level matching).
- [x] Fuzzy matching in BM25 with configurable edit distance (`max_edit_distance`).
- [x] Field-level BM25 boosts (weight title vs content vs tags differently in scoring).
- [x] Metadata aggregation / group-by (count/sum per attribute, e.g., hits per source).
- [x] Contextual retrieval: rewrite chunks with surrounding context before embedding.

## Maintenance

- [ ] Keep [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [README.md](README.md) aligned when behavior changes.
