# Architecture

## Overview

RAG is a Rust library and CLI for Retrieval-Augmented Generation (RAG) with support for multiple embedding models, vector stores, and interfaces (CLI, MCP, Library).

## Core Components

### 1. Embeddings Module (`src/embeddings.rs`)

**Purpose**: Generate vector embeddings from text using various models.

**Traits**:
- `EmbeddingModel`: Async trait for embedding generation
  - `embed(texts)`: Generate embeddings for multiple texts
  - `embed_single(text)`: Generate embedding for a single text

**Implementations**:
- `OpenAIEmbeddingModel`: Uses OpenAI's embedding API
- `OllamaEmbeddingModel`: Uses local Ollama models

**Usage**:
```rust
let model = OpenAIEmbeddingModel::new("api-key".to_string());
let embeddings = model.embed(vec!["text1", "text2"]).await?;
```

### 2. Vector Store Module (`src/vector_store.rs`)

**Purpose**: Store and retrieve documents with vector similarity search.

**Traits**:
- `VectorStore`: Async trait for vector storage operations
  - `add(document)`: Add a single document
  - `add_batch(documents)`: Add multiple documents
  - `search(query, top_k)`: Find similar documents
  - `search_with_filter(query, top_k, filter)`: Find similar documents with metadata filtering
  - `search_batch(queries, top_k)`: Batch search for multiple queries
  - `get(id)`: Retrieve document by ID
  - `delete(id)`: Remove document
  - `list(limit, offset)`: List documents with pagination
  - `count()`: Get total document count
  - `metric()`: Get the configured distance metric

**Types**:
- `Document`: Represents a stored document with content, metadata, and optional embedding
- `Similarity`: Contains a document and its similarity score
- `MetadataFilter`: Key-value metadata filtering
- `cosine_similarity()`: Helper function for computing similarity

**Implementations**:
- `InMemoryVectorStore`: Thread-safe in-memory store using `DashMap` and `FlatIndex`
- `MinimalVectorDB`: Simple in-memory store using `RwLock<HashMap>` and `FlatIndex`

### 3. Index Module (`src/index.rs`)

**Purpose**: Pluggable vector search indexes with multiple distance metrics.

**Traits**:
- `Index`: Core trait for vector search indexes
  - `add(document)`: Add document to index
  - `remove(id)`: Remove document from index
  - `search(query, top_k)`: Find top-k similar documents
  - `search_batch(queries, top_k)`: Parallel batch search
  - `clear()`: Remove all documents
  - `len()`: Number of indexed documents
  - `metric()`: Distance metric used

**Distance Metrics** (`DistanceMetric`):
- `Cosine`: Cosine similarity (default, best for text embeddings)
- `Euclidean`: Negative Euclidean distance (best for spatial data)
- `DotProduct`: Raw dot product (best for normalized vectors)
- `Manhattan`: Negative Manhattan/L1 distance

**Implementations**:
- `FlatIndex`: Brute-force exact search with parallel batch query support. Suitable for datasets < 100k documents.

**Utilities** (`utils`):
- `l2_normalize()`: In-place L2 normalization
- `l2_normalize_copy()`: Return a new normalized vector
- `validate_dimensions()`: Validate consistent vector dimensions in a batch

### 4. Chunker Module (`src/chunker.rs`)

**Purpose**: Split text into manageable chunks for embedding.

**Traits**:
- `TextChunker`: Trait for text chunking strategies
  - `chunk(text)`: Split text into chunks

**Implementations**:
- `FixedSizeChunker`: Fixed-size chunks with overlap
- `ParagraphChunker`: Split by paragraphs (double newlines)
- `SentenceChunker`: Split by sentences with max sentences per chunk

### 4. Retriever Module (`src/retriever.rs`)

**Purpose**: High-level interface combining embeddings, chunking, and retrieval.

**Methods**:
- `add_document(content)`: Add and embed a document
- `add_document_with_metadata(content, metadata)`: Add document with metadata
- `retrieve(query)`: Get relevant chunks
- `retrieve_with_scores(query)`: Get chunks with similarity scores
- `retrieve_filtered(query, filter)`: Get chunks with metadata filtering

### 5. MCP Module (`src/mcp.rs`)

**Purpose**: Model Context Protocol (MCP) server implementation.

**Components**:
- `McpServer`: Handles MCP protocol requests
- `McpRequest`: Incoming request structure
- `McpResponse`: Response structure
- `McpError`: Error structure

**Available Tools**:
- `rag_add_document`: Add documents to the store
- `rag_query`: Query for relevant documents
- `rag_list_documents`: List stored documents
- `rag_count`: Count total documents

### 6. Error Handling (`src/errors.rs`)

**Purpose**: Centralized error types and Result alias.

**Types**:
- `RagError`: Enum covering all possible errors
- `Result<T>`: Type alias for `Result<T, RagError>`

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    CLI / MCP / Library                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                      Retriever                          │
│  - Coordinates embeddings, chunking, and retrieval     │
└─────┬───────────────┬───────────────┬───────────────────┘
      │               │               │
      ▼               ▼               ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│ Embeddings  │ │ Chunker  │ │ Vector Store │
└─────────────┘ └──────────┘ └──────────────┘
      │                           │
      ▼                           ▼
┌─────────────┐           ┌──────────────┐
│  OpenAI     │           │   Index      │
│  Ollama     │           │  (pluggable) │
└─────────────┘           └──────────────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │  FlatIndex   │
                           │  (HNSW, etc) │
                           └──────────────┘
```

## Data Flow

### Adding a Document

1. User provides document content
2. Retriever chunks the text using configured chunker
3. Embedding model generates vectors for each chunk
4. Chunks with embeddings are stored in vector store
5. Document IDs are returned

### Querying

1. User provides a query
2. Embedding model generates query vector
3. Vector store delegates search to its `Index`
4. Index computes similarity using the configured `DistanceMetric`
5. Top-k results are returned with scores
6. Results can be filtered by metadata

## Extensibility

### Adding New Embedding Models

Implement the `EmbeddingModel` trait:

```rust
#[async_trait]
impl EmbeddingModel for MyEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Implementation
    }
}
```

### Adding New Indexes

Implement the `Index` trait for custom search algorithms:

```rust
impl Index for MyAnnIndex {
    fn add(&self, document: Document) {
        // Add to your ANN structure (HNSW, IVF, etc.)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<Similarity> {
        // Perform approximate nearest neighbor search
    }

    fn metric(&self) -> DistanceMetric {
        DistanceMetric::Cosine
    }
}
```

Then use it in a custom `VectorStore`:

```rust
struct AnnVectorStore {
    index: MyAnnIndex,
    documents: DashMap<String, Document>,
}

impl VectorStore for AnnVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        self.index.add(document.clone());
        self.documents.insert(document.id.clone(), document);
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Similarity>> {
        Ok(self.index.search(query, top_k))
    }
    // ... other methods
}
```

### Adding New Vector Stores

Implement the `VectorStore` trait:

```rust
#[async_trait]
impl VectorStore for MyVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        // Implementation
    }
    // ... other methods
}
```

### Adding New Chunkers

Implement the `TextChunker` trait:

```rust`
impl TextChunker for MyChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        // Implementation
    }
}
```

## Concurrency

- All public methods are `async` for non-blocking operations
- `InMemoryVectorStore` uses `DashMap` for thread-safe concurrent access
- Embedding requests are handled asynchronously via `tokio`

## Performance Considerations

1. **Batching**: The `embed` method processes multiple texts in a single request when possible. `search_batch` enables parallel query execution.
2. **Concurrent Search**: Vector store operations can be called concurrently via `DashMap` and parallel batch search.
3. **Index Strategy**: `FlatIndex` does exact brute-force search (O(n)). For large datasets (>100k), consider implementing HNSW or IVF via the `Index` trait.
4. **Memory**: In-memory store uses `DashMap` for efficient concurrent access without locking.
5. **Distance Metrics**: `Cosine` is default for text. `DotProduct` is faster if vectors are pre-normalized. `Euclidean`/`Manhattan` suit spatial data.
6. **Network**: Embedding requests are async and non-blocking

## Security

- API keys should be stored in environment variables
- No secrets are logged or exposed in error messages
- Input validation is performed on all user inputs

## Future Enhancements

1. Persistent vector stores (PostgreSQL pgvector, Qdrant, Pinecone)
2. Approximate nearest neighbor indexes (HNSW, IVF) via the `Index` trait
3. Additional embedding models (Cohere, HuggingFace)
4. Hybrid search (semantic + keyword BM25)
5. Document versioning
6. Cross-encoder reranking
7. Document deduplication