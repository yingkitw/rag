pub mod keyword;
pub mod hybrid;
pub mod dedup;
pub mod rerank;
pub mod index_ivf;
pub mod diversify;
pub mod aggregation;
pub mod fuzzy;
pub mod sparse;
#[cfg(feature = "http")]
pub mod rerank_external;
#[cfg(feature = "http")]
pub mod query_rewriting;
#[cfg(feature = "http")]
pub mod contextual;

pub mod embeddings;
pub mod vector_store;
pub mod retriever;
pub mod chunker;
pub mod errors;
#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "sqlite")]
pub mod store_sqlite;
pub mod index;
#[cfg(feature = "hnsw")]
pub mod index_hnsw;
pub mod ingestion;
pub mod graph;
pub mod graph_rag;

pub use embeddings::EmbeddingModel;
#[cfg(feature = "http")]
pub use embeddings::{HttpEmbeddingModel, OpenAIEmbeddingModel, EmbeddingRequest, OllamaEmbeddingModel};
pub use vector_store::{
    JsonPersistentVectorStore, VectorStore, InMemoryVectorStore, MinimalVectorDB, Document,
    Similarity, MetadataFilter, load_all_documents,
};
pub use retriever::Retriever;
pub use chunker::{TextChunker, FixedSizeChunker, ParagraphChunker, SentenceChunker};
pub use errors::{RagError, Result};
#[cfg(feature = "mcp")]
pub use mcp::RagMcpServer;
pub use index::{DistanceMetric, FlatIndex, Index};
#[cfg(feature = "hnsw")]
pub use index_hnsw::HnswIndex;
pub use index_ivf::IvfflatIndex;
#[cfg(feature = "sqlite")]
pub use store_sqlite::SqliteVectorStore;
pub use ingestion::{Source, ExtractedDocument};
#[cfg(feature = "pdf")]
pub use ingestion::PdfSource;
#[cfg(feature = "ingest")]
pub use ingestion::CodebaseSource;
#[cfg(feature = "http")]
pub use ingestion::WikiSource;
pub use graph::{GraphStore, GraphNode, GraphEdge, GraphPath, Community, GraphPersisted};
pub use graph_rag::{
    EntityExtractor, ExtractedEntity, GraphInfo, GraphRagEngine, GraphRagResult, GraphRagSnapshot,
    SeedEntityExtractor, SimpleEntityExtractor, EntityInfo,
};
#[cfg(feature = "llm-extractor")]
pub use graph_rag::LlmEntityExtractor;
pub use keyword::{tokenize, Bm25Index, Bm25Config, FieldBm25Index};
pub use hybrid::{merge_hybrid, rrf_fusion};
pub use dedup::{content_jaccard, dedup_similarities};
pub use rerank::{SimilarityReranker, PassthroughReranker, rerank_similarities};
pub use diversify::diversify;
pub use aggregation::{count_by, group_by, sum_by};
pub use fuzzy::{levenshtein, is_fuzzy_match, fuzzy_filter};
pub use sparse::{SparseVector, SparseIndex};
#[cfg(feature = "http")]
pub use rerank_external::{CohereReranker, VoyageReranker, MixedBreadReranker};
#[cfg(feature = "http")]
pub use query_rewriting::QueryRewriter;
#[cfg(feature = "http")]
pub use contextual::ContextualRetrieval;
