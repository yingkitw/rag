# Architecture

## Overview

This repository is a Rust **library**, **CLI** (`rag`), and **MCP server** (`rag-mcp-server`) for Retrieval-Augmented Generation. Retrieval is deliberately **multi-signal**:

- **Vectors** — semantic similarity over embedded chunks.
- **Graph** — entities and co-occurrence (and similar) edges for expansion and structure.
- **Search-style operations** — top-k limits, scored ranking, listing, counting, and metadata filters.

The vector-centric entry point is `Retriever`. The combined vector + graph path is `GraphRagEngine` (library) and a parallel set of MCP tools under the `graph_*` namespace.

## Core components

### 1. Embeddings (`src/embeddings.rs`)

**Purpose:** Turn text into dense vectors.

**Trait:**

- `EmbeddingModel`: `embed`, `embed_single`.

**Implementations:**

- `OpenAIEmbeddingModel`
- `OllamaEmbeddingModel`

### 2. Vector store (`src/vector_store.rs`)

**Purpose:** Persist chunks and run similarity search.

**Trait `VectorStore`:** `add`, `add_batch`, `search`, `search_with_filter`, `search_batch`, `get`, `delete`, `list`, `count`, `metric`.

**Types:** `Document`, `Similarity`, `MetadataFilter`.

**Implementations:**

- `InMemoryVectorStore` — `DashMap` + pluggable `Index`.
- `MinimalVectorDB` — `RwLock` map + `FlatIndex`.

### 3. Index (`src/index.rs`)

**Purpose:** Pluggable nearest-neighbor logic and distance metrics.

**Trait `Index`:** `add`, `remove`, `search`, `search_batch`, `clear`, `len`, `metric`.

**Metrics:** `Cosine`, `Euclidean`, `DotProduct`, `Manhattan`.

**Implementation:** `FlatIndex` — exact, parallel batch queries; suitable for modest corpus sizes.

### 4. Chunker (`src/chunker.rs`)

**Purpose:** Split documents before embedding.

**Trait:** `TextChunker::chunk`.

**Implementations:** `FixedSizeChunker`, `ParagraphChunker`, `SentenceChunker`.

### 5. Retriever (`src/retriever.rs`)

**Purpose:** Orchestrate chunking, embedding, and vector search for classic RAG.

**Methods:** `add_document`, `add_document_with_metadata`, `retrieve`, `retrieve_with_scores`, `retrieve_filtered`.

### 6. Graph (`src/graph.rs`)

**Purpose:** In-memory property graph for RAG augmentation.

**Types:** `GraphNode`, `GraphEdge`, `GraphPath`, `Community`.

**`GraphStore`:** add/remove nodes and edges, lookup by id or name, BFS-style reachability, neighbors, degree, density, community detection (label propagation), optional save/load helpers where implemented.

### 7. Graph RAG (`src/graph_rag.rs`)

**Purpose:** Single engine that writes to both `VectorStore` and `GraphStore`.

**Pieces:**

- `EntityExtractor` / `SimpleEntityExtractor` — heuristic entities (quoted strings, acronyms, proper nouns).
- `GraphRagEngine` — `add_document` (chunk, embed, store chunk, link co-occurring entities), `query` — merges vector top-k with chunks linked to entities in the query neighborhood (`graph_depth`, `top_k`).

**Types:** `GraphRagResult`, `EntityInfo`, `GraphInfo` (and related) for structured results.

### 8. Ingestion (`src/ingestion.rs`)

**Purpose:** Pull text from external sources into `ExtractedDocument` records for downstream chunking and embedding.

**Trait:** `Source::extract`.

**Sources:** `PdfSource`, `CodebaseSource`, `WikiSource`, and related helpers.

### 9. MCP (`src/mcp.rs`, binary `src/mcp_server.rs`)

**Purpose:** Expose RAG operations over Model Context Protocol (stdio transport, `rmcp`).

**Handler:** `RagMcpServer` — shared `InMemoryVectorStore`, `GraphStore`, embedding backend (`OpenAI` or `Ollama`), `SimpleEntityExtractor`, and side maps from entity names to chunk ids and reverse.

**Vector tools:**

- `rag_add_document` — chunk, embed, index.
- `rag_query` — semantic search with scores.
- `rag_list_documents` — paginated listing.
- `rag_count` — chunk count.

**Graph and hybrid tools:**

- `graph_build` — like add, but also extracts entities, links co-occurrence, stores entity metadata on chunks, updates graph.
- `graph_query` — vector search plus graph expansion from query entities; labels hits as `vector` vs `graph`.
- `graph_get_entity`, `graph_get_neighbors` — introspection.
- `graph_info`, `graph_communities` — stats and community structure.

### 10. Errors (`src/errors.rs`)

**Purpose:** `RagError` and `Result<T>` alias for unified error handling.

## Architecture diagram

```
                    ┌──────────────────────────────────────┐
                    │   CLI (`rag`)  /  Library  /  MCP     │
                    └────────────────────┬─────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           ▼                             ▼                             ▼
    ┌─────────────┐              ┌──────────────┐              ┌─────────────┐
    │  Retriever  │              │ GraphRagEngine│             │ RagMcpServer │
    │ (vector RAG)│              │ (vector+graph)│             │ (both tools) │
    └──────┬──────┘              └───────┬───────┘              └──────┬──────┘
           │                            │                             │
           │              ┌─────────────┴─────────────┐               │
           │              ▼                           ▼               │
           │       ┌────────────┐            ┌────────────┐           │
           │       │ GraphStore │            │EntitySide │◄──────────┤
           │       │ nodes/edges│            │  maps     │           │
           │       └────────────┘            └────────────┘           │
           │                                                          │
           ▼                            ▼                             ▼
    ┌─────────────┐              ┌──────────────┐              ┌──────────────┐
    │ Embeddings  │              │  Chunker      │             │  Chunker     │
    │ OpenAI/     │              │ (strategies)  │             │ (MCP paths)  │
    │ Ollama      │              └──────────────┘             └──────────────┘
    └──────┬──────┘                      │                             │
           │                              ▼                             ▼
           │                       ┌──────────────┐             ┌──────────────┐
           └──────────────────────►│ VectorStore   │◄────────────┤ InMemory*    │
                                     │ + Index      │             │ + FlatIndex  │
                                     └──────────────┘             └──────────────┘

    Ingestion:  PdfSource / CodebaseSource / WikiSource  ──► ExtractedDocument ──► Retriever or GraphRagEngine
```

## Data flow

### Add document (Retriever)

1. Chunk text.
2. Embed chunks.
3. Store in `VectorStore` (index updated).

### Add document with graph (`GraphRagEngine` or MCP `graph_build`)

1. Chunk and embed as above.
2. Extract entities per chunk; ensure nodes in `GraphStore`.
3. Add co-occurrence edges between entities in the same chunk (with weights where supported).
4. Record entity-to-chunk and chunk-to-entity mappings.
5. Persist chunks in `VectorStore`.

### Hybrid query (`GraphRagEngine::query` or MCP `graph_query`)

1. Embed query; run vector `search` for top-k.
2. Extract query entities; traverse graph to a configured depth.
3. Collect chunk ids linked to matched entities; merge with vector results (deduplicate, cap at top-k).
4. Return ranked list with scores where available and provenance (`vector` vs `graph` in MCP JSON).

## Extensibility

### New embedding models

Implement `EmbeddingModel` with `async_trait` and delegate to your provider.

### New indexes

Implement `Index` (for example HNSW) and use it inside a custom `VectorStore` or swap the index inside `InMemoryVectorStore` if the API allows.

### New vector stores

Implement `VectorStore` for disk or remote backends.

### New chunkers

Implement `TextChunker`:

```rust
impl TextChunker for MyChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        // ...
    }
}
```

### Richer graphs

Implement `EntityExtractor` to feed `GraphRagEngine` with NER or model-based entities.

## Concurrency

- Public APIs are async (`tokio`).
- `InMemoryVectorStore` and `GraphStore` use `DashMap` for concurrent access.
- Batch index search can run work in parallel depending on `FlatIndex` usage.

## Performance

- `FlatIndex` is O(n) per query; plan an ANN `Index` for large n (see [TODO.md](TODO.md)).
- Prefer batch embedding where the model allows.
- Normalize vectors when using dot-product metric for stability.

## Security

- Read API keys from environment variables; avoid printing them.
- Treat ingested paths and URLs as untrusted; validate and sandbox as needed in calling code.

## Related docs

- [SPEC.md](SPEC.md) — scope and requirements.
- [TODO.md](TODO.md) — planned enhancements.
