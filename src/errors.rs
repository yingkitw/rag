use thiserror::Error;

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Embedding API error: {0}")]
    EmbeddingError(String),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Vector store error: {0}")]
    VectorStoreError(String),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, RagError>;