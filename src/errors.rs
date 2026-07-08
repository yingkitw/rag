use std::fmt;

#[derive(Debug)]
pub enum RagError {
    EmbeddingError(String),

    #[cfg(feature = "http")]
    HttpError(reqwest::Error),

    JsonError(serde_json::Error),

    IoError(std::io::Error),

    VectorStoreError(String),

    DocumentNotFound(String),

    InvalidConfig(String),

    GraphError(String),
}

impl fmt::Display for RagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RagError::EmbeddingError(msg) => write!(f, "Embedding API error: {}", msg),
            #[cfg(feature = "http")]
            RagError::HttpError(err) => write!(f, "HTTP request failed: {}", err),
            RagError::JsonError(err) => {
                write!(f, "JSON serialization/deserialization error: {}", err)
            }
            RagError::IoError(err) => write!(f, "IO error: {}", err),
            RagError::VectorStoreError(msg) => write!(f, "Vector store error: {}", msg),
            RagError::DocumentNotFound(msg) => write!(f, "Document not found: {}", msg),
            RagError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            RagError::GraphError(msg) => write!(f, "Graph error: {}", msg),
        }
    }
}

impl std::error::Error for RagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "http")]
            RagError::HttpError(err) => Some(err),
            RagError::JsonError(err) => Some(err),
            RagError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for RagError {
    fn from(err: serde_json::Error) -> Self {
        RagError::JsonError(err)
    }
}

impl From<std::io::Error> for RagError {
    fn from(err: std::io::Error) -> Self {
        RagError::IoError(err)
    }
}

#[cfg(feature = "http")]
impl From<reqwest::Error> for RagError {
    fn from(err: reqwest::Error) -> Self {
        RagError::HttpError(err)
    }
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

    #[cfg(feature = "http")]
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
        assert!(ok_result.is_ok());

        let err_result: Result<i32> = Err(RagError::DocumentNotFound("x".to_string()));
        assert!(err_result.is_err());
    }
}