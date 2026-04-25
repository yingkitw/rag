# Quick Reference Guide

## CLI Usage

### Setup
```bash
# Set OpenAI API key (optional, will use Ollama if not set)
export OPENAI_API_KEY="your-key"

# Or use Ollama (default: localhost:11434)
export OLLAMA_URL="http://localhost:11434"
```

### Commands

#### Add Documents
```bash
# Add a single file
rag add --file document.txt --source "my-docs"

# The command will:
# 1. Read the file
# 2. Split into chunks
# 3. Generate embeddings
# 4. Store in vector database
```

#### Query Documents
```bash
# Search for relevant content
rag query --query "What is Rust?" --top-k 3

# Output shows:
# 1. Similarity score
# 2. Chunk content
```

#### List Documents
```bash
# List first 10 documents
rag list --limit 10

# List with offset
rag list --limit 5 --offset 10
```

#### Count Documents
```bash
rag count
```

## Library Usage

### Basic Example

```rust
use rag::{
    chunker::FixedSizeChunker,
    embeddings::OpenAIEmbeddingModel,
    retriever::Retriever,
    vector_store::InMemoryVectorStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create components
    let embedding_model = OpenAIEmbeddingModel::new("api-key".to_string());
    let vector_store = InMemoryVectorStore::new();
    
    // 2. Create retriever
    let retriever = Retriever::new(embedding_model, vector_store)
        .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
        .with_top_k(5);
    
    // 3. Add document
    retriever.add_document("Your text here".to_string()).await?;
    
    // 4. Query
    let results = retriever.retrieve("search query".to_string()).await?;
    
    Ok(())
}
```

### With Metadata

```rust
let retriever = Retriever::new(embedding_model, vector_store)
    .with_top_k(5);

retriever.add_document_with_metadata(
    "Document content".to_string(),
    vec![
        ("source".to_string(), "document.pdf".to_string()),
        ("author".to_string(), "John Doe".to_string()),
        ("date".to_string(), "2024-01-01".to_string()),
    ]
).await?;

// Retrieve with metadata filter
let results = retriever.retrieve_filtered("query", "John Doe").await?;
```

### Different Chunking Strategies

```rust
use rag::chunker::{FixedSizeChunker, ParagraphChunker, SentenceChunker};

// Fixed-size with overlap
let chunker = Box::new(FixedSizeChunker::new(500, 50));

// By paragraphs
let chunker = Box::new(ParagraphChunker);

// By sentences
let chunker = Box::new(SentenceChunker::new(5));

let retriever = Retriever::new(embedding_model, vector_store)
    .with_chunker(chunker)
    .with_top_k(5);
```

### Using Ollama

```rust
use rag::embeddings::OllamaEmbeddingModel;

let model = OllamaEmbeddingModel::new("nomic-embed-text".to_string())
    .with_base_url("http://localhost:11434".to_string());
```

## MCP Server

### Start MCP Server
```bash
cargo run --bin rag-mcp-server
```

### MCP Configuration
Add to your MCP client config:

```json
{
  "mcpServers": {
    "rag": {
      "command": "cargo",
      "args": [
        "run",
        "--bin",
        "rag-mcp-server",
        "--manifest-path",
        "/path/to/rag/Cargo.toml"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### MCP Tools

#### Add Document
```json
{
  "name": "rag_add_document",
  "arguments": {
    "content": "Document text",
    "source": "optional-source"
  }
}
```

#### Query
```json
{
  "name": "rag_query",
  "arguments": {
    "query": "search query",
    "top_k": 5
  }
}
```

#### List Documents
```json
{
  "name": "rag_list_documents",
  "arguments": {
    "limit": 10,
    "offset": 0
  }
}
```

#### Count
```json
{
  "name": "rag_count",
  "arguments": {}
}
```

## Common Patterns

### Process Multiple Files
```rust
use tokio::fs;

async fn process_directory(path: &str, retriever: &Retriever<_, _>) -> Result<()> {
    let mut entries = fs::read_dir(path).await?;
    
    while let Some(entry) = entries.next_entry().await? {
        if entry.path().extension().map_or(false, |e| e == "txt") {
            let content = fs::read_to_string(entry.path()).await?;
            retriever.add_document_with_metadata(
                content,
                vec![("file".to_string(), entry.path().to_string_lossy().into())],
            ).await?;
        }
    }
    
    Ok(())
}
```

### Batch Queries
```rust
let queries = vec!["query 1", "query 2", "query 3"];
let mut results = Vec::new();

for query in queries {
    let query_results = retriever.retrieve_with_scores(query).await?;
    results.push((query, query_results));
}
```

### Custom Error Handling
```rust
use rag::errors::RagError;

match retriever.add_document(content).await {
    Ok(_) => println!("Document added"),
    Err(RagError::EmbeddingError(e)) => eprintln!("Embedding failed: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None (uses Ollama) |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `RUST_LOG` | Logging level | `info` |

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_vector_store

# Run examples
cargo run --example simple_rag
cargo run --example mcp_example
```

## Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with specific features
cargo build --features ollama
```

## Troubleshooting

### Connection Errors
- Check your API key
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check network connectivity

### Memory Issues
- Reduce chunk size
- Process documents in batches
- Use persistent vector store (future feature)

### Slow Performance
- Use batch embedding when possible
- Reduce top_k for queries
- Consider using faster embedding models