# RAG

A Rust library and CLI tool for Retrieval-Augmented Generation (RAG) with support for multiple embedding models and vector stores.

## Features

- 🦀 Pure Rust implementation with async/await support
- 🤖 Multiple embedding model support (OpenAI, Ollama)
- 📊 In-memory vector stores with cosine similarity search (DashMap and minimal implementations)
- 📝 Multiple text chunking strategies (fixed-size, paragraph, sentence)
- 🎯 Configurable top-k retrieval
- 🔍 Metadata filtering support
- 💻 CLI tool for quick operations
- 📚 Easy-to-use library API

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

See the `examples/` directory for more usage examples:

```bash
cargo run --example simple_rag
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, will use Ollama if not set)
- `OLLAMA_URL`: Ollama server URL (default: `http://localhost:11434`)

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
- `Retriever`: Main interface for RAG operations
- `Document`: Represents a stored document with content, metadata, and optional embedding
- `TextChunker`: Trait for text chunking strategies

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
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.