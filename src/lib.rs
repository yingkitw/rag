pub mod aggregation;
#[cfg(feature = "compress")]
pub mod compress;
#[cfg(feature = "http")]
pub mod contextual;
pub mod dedup;
pub mod diversify;
pub mod eval;
#[cfg(feature = "fastembed")]
pub mod fastembed_store;
pub mod fuzzy;
pub mod hybrid;
pub mod id;
pub mod index_ivf;
pub mod keyword;
pub mod quantize;
#[cfg(feature = "http")]
pub mod query_rewriting;
pub mod rerank;
#[cfg(feature = "http")]
pub mod rerank_external;
pub mod simd;
pub mod sparse;
pub mod wal;

pub mod chunker;
pub mod embeddings;
pub mod errors;
pub mod graph;
pub mod graph_rag;
pub mod index;
#[cfg(feature = "hnsw")]
pub mod index_hnsw;
pub mod ingestion;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod retriever;
#[cfg(feature = "postgres")]
pub mod store_postgres;
#[cfg(feature = "qdrant")]
pub mod store_qdrant;
#[cfg(feature = "sqlite")]
pub mod store_sqlite;
pub mod vector_store;

pub use aggregation::{count_by, group_by, sum_by};
pub use chunker::{FixedSizeChunker, ParagraphChunker, SentenceChunker, TextChunker};
pub use chunker::{RecursiveChunker, SemanticChunker, StructuralChunker};
#[cfg(feature = "compress")]
pub use compress::{compress_bytes, decompress_bytes, load_compressed, save_compressed};
#[cfg(feature = "http")]
pub use contextual::ContextualRetrieval;
pub use dedup::{content_jaccard, dedup_similarities};
pub use diversify::diversify;
pub use embeddings::EmbeddingModel;
#[cfg(feature = "http")]
pub use embeddings::{
    EmbeddingRequest, HttpEmbeddingModel, OllamaEmbeddingModel, OpenAIEmbeddingModel,
};
pub use errors::{RagError, Result};
pub use eval::{
    EvalReport, average_precision, dcg_at_k, evaluate, graded_relevance, idcg_at_k, ndcg_at_k,
    precision_at_k, recall_at_k, reciprocal_rank, relevance_set,
};
#[cfg(feature = "image-embeddings")]
pub use fastembed_store::FastEmbedImageEmbeddingModel;
#[cfg(feature = "fastembed")]
pub use fastembed_store::{FastEmbedEmbeddingModel, FastEmbedReranker};
pub use fuzzy::{fuzzy_filter, is_fuzzy_match, levenshtein};
pub use graph::{Community, GraphEdge, GraphNode, GraphPath, GraphPersisted, GraphStore};
#[cfg(feature = "llm-extractor")]
pub use graph_rag::LlmEntityExtractor;
pub use graph_rag::{
    EntityExtractor, EntityInfo, ExtractedEntity, GraphInfo, GraphRagEngine, GraphRagResult,
    GraphRagSnapshot, SeedEntityExtractor, SimpleEntityExtractor,
};
pub use hybrid::{merge_hybrid, rrf_fusion};
pub use index::{DistanceMetric, FlatIndex, Index};
#[cfg(feature = "hnsw")]
pub use index_hnsw::HnswIndex;
pub use index_ivf::IvfflatIndex;
#[cfg(feature = "ingest")]
pub use ingestion::CodebaseSource;
#[cfg(feature = "pdf")]
pub use ingestion::PdfSource;
#[cfg(feature = "http")]
pub use ingestion::WikiSource;
pub use ingestion::{ExtractedDocument, Source};
pub use keyword::{Bm25Config, Bm25Index, FieldBm25Index, tokenize};
#[cfg(feature = "mcp")]
pub use mcp::RagMcpServer;
pub use quantize::{QuantizationParams, QuantizedIndex};
#[cfg(feature = "http")]
pub use query_rewriting::QueryRewriter;
pub use rerank::{PassthroughReranker, SimilarityReranker, rerank_similarities};
#[cfg(feature = "http")]
pub use rerank_external::{CohereReranker, MixedBreadReranker, VoyageReranker};
pub use retriever::Retriever;
pub use sparse::{SparseIndex, SparseVector};
#[cfg(feature = "postgres")]
pub use store_postgres::PostgresVectorStore;
#[cfg(feature = "qdrant")]
pub use store_qdrant::QdrantVectorStore;
#[cfg(feature = "sqlite")]
pub use store_sqlite::SqliteVectorStore;
pub use vector_store::{
    Document, InMemoryVectorStore, JsonPersistentVectorStore, MetadataFilter, MinimalVectorDB,
    Similarity, VectorStore, load_all_documents,
};
pub use wal::{WalOp, WriteAheadLog, apply_ops};
