# RAG

A Rust library and CLI for Retrieval-Augmented Generation (RAG) that combines vector similarity, graph structure, and search-style retrieval rather than embeddings alone. Dense vectors cover semantic match, a knowledge graph encodes entities and relations, and configurable top-k plus metadata filtering make retrieval behave like a search layer over your corpus.

Project docs: [SPEC.md](SPEC.md) (scope and requirements), [ARCHITECTURE.md](ARCHITECTURE.md) (modules and data flow), [TODO.md](TODO.md) (backlog).

## Features

- Pure Rust implementation with async/await support
- Vector RAG: multiple embedding backends (OpenAI, Ollama), pluggable indexes and distance metrics (cosine, Euclidean, dot product, Manhattan)
- Graph RAG: graph store for nodes and edges, entity extraction hooks, and a `GraphRagEngine` that ties documents, vectors, and the graph together
- In-memory vector stores with parallel batch search (`InMemoryVectorStore`, `MinimalVectorDB`)
- Search-oriented retrieval: configurable top-k, score-ranked results, and metadata filtering over stored chunks
- Ingestion helpers: `Source` implementations for PDF, codebase trees, and wiki-style URLs (`ingestion` module)
- Multiple text chunking strategies (fixed-size, paragraph, sentence)
- CLI for ingest and query with **persistent state** (`RAG_STATE_DIR`, default `.rag`): vector, **hybrid-query (BM25 + embeddings)**, and **graph** subcommands
- MCP server (`rag-mcp`) with vector tools (`rag_*`) and graph or hybrid tools (`graph_*`)
- Library API suitable for custom pipelines

## Installation

### From source

```bash
cargo install --path .
```

### As a library

Add to your `Cargo.toml`:

```toml
[dependencies]
rag = { git = "https://github.com/yingkitw/rag" }
```

## Quick Start

State for the CLI lives under **`RAG_STATE_DIR`** (default `.rag`): `vectors.json`, optional `graph.json` and `graph_rag.json`.

### CLI Usage

```bash
# Set your API key (OpenAI) or use Ollama
export OPENAI_API_KEY="your-api-key-here"
# Optional when using Ollama for CLI or rag-mcp:
export OLLAMA_MODEL="nomic-embed-text"

# Add a document (persists chunks to $RAG_STATE_DIR/vectors.json)
rag add --file document.txt --source "my-docs"

# Add multiple files
rag add --file a.txt --file b.md --source "batch"

# Add all .txt / .md from a directory
rag add --file ./docs/ --source "wiki"

# Vector-only query
rag query --query "What is Rust?" --top-k 3

# Vector + BM25 hybrid (alpha = vector weight in [0,1])
rag hybrid-query --query "What is Rust?" --top-k 5 --alpha 0.65

# Change chunker or distance metric at runtime
rag query --query "What is Rust?" --chunker sentence --metric euclidean

# Graph stats from a saved graph file
rag graph-stats

# Build GraphRAG snapshot from a file (writes graph_rag.json + graph.json)
# Subsequent runs merge into the existing snapshot (incremental)
rag graph-build --file document.txt --source "my-docs"

# Build another document into the same snapshot
rag graph-build --file another.txt --source "more-docs"

# Query using saved GraphRAG snapshot
rag graph-hybrid-query --query "Who is mentioned?" --top-k 5

# List documents
rag list --limit 10 --offset 0

# Count documents
rag count
```

### Library Usage

```rust
use rag::{
    chunker::FixedSizeChunker,
    embeddings::OpenAIEmbeddingModel,
    retriever::Retriever,
    vector_store::MinimalVectorDB,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedding model and vector store
    let embedding_model = OpenAIEmbeddingModel::new("your-api-key".to_string());
    let vector_store = MinimalVectorDB::new();
    
    // Create retriever
    let retriever = Retriever::new(embedding_model, vector_store)
        .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
        .with_top_k(5);
    
    // Add documents
    retriever.add_document("Your document content here".to_string()).await?;
    
    // Retrieve relevant chunks
    let results = retriever.retrieve("Your query here").await?;
    
    for (i, content) in results.iter().enumerate() {
        println!("{}. {}", i + 1, content);
    }
    
    Ok(())
}
```

## Examples

See the `examples/` directory:

```bash
cargo run --example simple_rag
cargo run --example pure_memory_rag
cargo run --example advanced_vector_store
cargo run --example minimal_vector_db
cargo run --example batch_search
cargo run --example distance_metrics
cargo run --example graph_store_basic
cargo run --example graph_rag_example
cargo run --example ingest_fixture_rag
cargo run --example ingest_pdf
cargo run --example ingest_codebase
cargo run --example ingest_wiki
cargo run --example mcp_example
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional; if unset, embeddings use Ollama)
- `OLLAMA_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Embedding model when using **Ollama** (CLI, `rag-mcp`, and examples; default: `nomic-embed-text`)

### MCP server

Run the stdio MCP server (for clients that spawn the process):

```bash
export OPENAI_API_KEY="..."   # or rely on Ollama + OLLAMA_URL / OLLAMA_MODEL
cargo run --bin rag-mcp
```

Vector tools: `rag_add_document`, `rag_query`, `rag_list_documents`, `rag_count`. Graph and hybrid tools: `graph_build`, `graph_query`, `graph_get_entity`, `graph_get_neighbors`, `graph_info`, `graph_communities`.

### CLI Global Flags

- `--chunker <fixed|paragraph|sentence>`: Chunking strategy (default: `paragraph`)
- `--metric <cosine|euclidean|dot|manhattan>`: Distance metric for vector search (default: `cosine`)
- `--state-dir <path>`: State directory (default: `.rag`; also set via `RAG_STATE_DIR`)

### Entity Extraction

- `SimpleEntityExtractor`: Rule-based extractor (acronyms, quoted terms, proper nouns)
- `SeedEntityExtractor`: Match a fixed list of seed entities in text
- `LlmEntityExtractor` (requires `llm-extractor` feature): Uses an LLM for high-quality NER

Enable the LLM extractor:

```bash
cargo build --features llm-extractor
```

```rust
#[cfg(feature = "llm-extractor")]
use rag::LlmEntityExtractor;

let extractor = LlmEntityExtractor::new("your-openai-key".to_string());
let engine = GraphRagEngine::new(extractor, embedding_model, store);
```

### Vector Indexes

- `FlatIndex`: Exact brute-force search (best for small datasets, < 100k docs)
- `IvfflatIndex`: IVF (Inverted File) index — first ANN step, faster than flat at scale
- `HnswIndex`: HNSW (Hierarchical Navigable Small World) approximate index using `hnsw_rs` — best for large datasets where approximate recall is acceptable

```rust
use rag::{HnswIndex, Index, DistanceMetric};

let index = HnswIndex::with_metric(DistanceMetric::Cosine);
index.add(doc);
let results = index.search(&query_embedding, 10);
```

### Chunking Strategies

- `FixedSizeChunker`: Splits text into chunks of fixed size with overlap
- `ParagraphChunker`: Splits text by paragraphs (double newlines)
- `SentenceChunker`: Splits text by sentences

### Embedding Models

#### OpenAI
```rust
let model = OpenAIEmbeddingModel::new("your-api-key".to_string());
let model = OpenAIEmbeddingModel::with_model("your-api-key".to_string(), "text-embedding-ada-002".to_string());
```

#### Ollama
```rust
let model = OllamaEmbeddingModel::new("nomic-embed-text".to_string());
let model = OllamaEmbeddingModel::new("nomic-embed-text".to_string())
    .with_base_url("http://localhost:11434".to_string());
```

## API Reference

### Core Types

- `EmbeddingModel`: Trait for embedding models
- `VectorStore`: Trait for vector storage backends
- `Retriever`: Main interface for vector-centric RAG operations
- `GraphStore`, `GraphNode`, `GraphEdge`: Graph storage and structure for graph-augmented retrieval
- `GraphRagEngine`, `EntityExtractor`: Orchestration and entity linking for graph RAG
- `Source`, `ExtractedDocument`: Ingestion from PDF, codebase, wiki, and other sources
- `Document`: Represents a stored document with content, metadata, and optional embedding
- `TextChunker`: Trait for text chunking strategies
- `Index`: Trait for vector search indexes (`FlatIndex`, `IvfflatIndex`, `HnswIndex`)
- `RagMcpServer`: MCP tool router combining vector store and graph (see `mcp` module)

### Retriever Methods

- `add_document(content)`: Add a single document
- `add_document_with_metadata(content, metadata)`: Add a document with metadata
- `retrieve(query)`: Retrieve relevant chunks
- `retrieve_with_scores(query)`: Retrieve chunks with similarity scores
- `retrieve_filtered(query, metadata_filter)`: Retrieve with metadata filtering

## Development

Run tests:

```bash
cargo test
```

Run examples:

```bash
cargo run --example simple_rag
cargo run --example pure_memory_rag
cargo run --example graph_store_basic
cargo run --example graph_rag_example
cargo run --example ingest_fixture_rag
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
