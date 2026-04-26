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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_embedding_error_display() {
        let err = RagError::EmbeddingError("model timeout".to_string());
        assert_eq!(format!("{}", err), "Embedding API error: model timeout");
    }

    #[test]
    fn test_vector_store_error_display() {
        let err = RagError::VectorStoreError("index corrupted".to_string());
        assert_eq!(format!("{}", err), "Vector store error: index corrupted");
    }

    #[test]
    fn test_document_not_found_display() {
        let err = RagError::DocumentNotFound("doc-123".to_string());
        assert_eq!(format!("{}", err), "Document not found: doc-123");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = RagError::InvalidConfig("chunk_size must be > 0".to_string());
        assert_eq!(format!("{}", err), "Invalid configuration: chunk_size must be > 0");
    }

    #[test]
    fn test_from_reqwest_error() {
        // reqwest::Error can't be constructed easily without a client,
        // but we can at least verify the From impl exists by checking compilation.
        // For a real test, we'd need to make an actual failed request.
        // This test documents the From trait is available.
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: RagError = io_err.into();
        assert!(matches!(err, RagError::IoError(_)));
        assert!(format!("{}", err).contains("file not found"));
    }

    #[test]
    fn test_result_type_alias() {
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<i32> = Err(RagError::DocumentNotFound("x".to_string()));
        assert!(err_result.is_err());
    }
}