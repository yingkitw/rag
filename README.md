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
- CLI for quick ingest and query operations (`rag` binary)
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

### CLI Usage

```bash
# Set your API key (OpenAI or use Ollama)
export OPENAI_API_KEY="your-api-key-here"

# Add a document
rag add --file document.txt --source "my-docs"

# Query the vector store
rag query --query "What is Rust?" --top-k 3

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

See the `examples/` directory, for example:

```bash
cargo run --example simple_rag
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
- `OLLAMA_MODEL`: Embedding model name when using Ollama with `rag-mcp-server` (default: `nomic-embed-text`)

### MCP server

Run the stdio MCP server (for clients that spawn the process):

```bash
export OPENAI_API_KEY="..."   # or rely on Ollama + OLLAMA_URL / OLLAMA_MODEL
cargo run --bin rag-mcp
```

Vector tools: `rag_add_document`, `rag_query`, `rag_list_documents`, `rag_count`. Graph and hybrid tools: `graph_build`, `graph_query`, `graph_get_entity`, `graph_get_neighbors`, `graph_info`, `graph_communities`.

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
cargo run --example graph_store_basic
cargo run --example graph_rag_example
cargo run --example ingest_fixture_rag
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
