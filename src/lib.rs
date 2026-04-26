pub mod embeddings;
pub mod vector_store;
pub mod retriever;
pub mod chunker;
pub mod errors;
pub mod mcp;
pub mod index;
pub mod ingestion;

pub use embeddings::{EmbeddingModel, OpenAIEmbeddingModel, EmbeddingRequest, OllamaEmbeddingModel};
pub use vector_store::{VectorStore, InMemoryVectorStore, MinimalVectorDB, Document, Similarity, MetadataFilter};
pub use retriever::Retriever;
pub use chunker::{TextChunker, FixedSizeChunker, ParagraphChunker, SentenceChunker};
pub use errors::{RagError, Result};
pub use mcp::{McpServer, McpRequest, McpResponse};
pub use index::{DistanceMetric, FlatIndex, Index};
pub use ingestion::{Source, ExtractedDocument, PdfSource, CodebaseSource, WikiSource};