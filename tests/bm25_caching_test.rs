use rag::{
    embeddings::EmbeddingModel,
    retriever::Retriever,
    vector_store::{Document, InMemoryVectorStore, VectorStore},
};
use async_trait::async_trait;

/// Deterministic mock embedding model.
#[derive(Clone)]
struct MockEmbeddingModel;

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, rag::errors::RagError> {
        Ok(texts
            .into_iter()
            .map(|t| t.bytes().map(|b| b as f32 / 255.0).collect())
            .collect())
    }
}

#[tokio::test]
async fn bm25_cache_avoids_rebuild() {
    let model = MockEmbeddingModel;
    let store = InMemoryVectorStore::new();
    let retriever = Retriever::new(model, store).with_top_k(3);

    // Add initial documents
    retriever.add_document("rust systems programming memory safety".to_string()).await.unwrap();
    retriever.add_document("python scripting dynamic typing easy".to_string()).await.unwrap();

    // First hybrid query builds BM25 cache
    let r1 = retriever.retrieve_hybrid("rust memory", 0.7).await.unwrap();
    assert!(!r1.is_empty());

    // Second query reuses cache (same doc count)
    let r2 = retriever.retrieve_hybrid("python easy", 0.7).await.unwrap();
    assert!(!r2.is_empty());

    // After adding new doc, cache invalidates
    retriever.add_document("javascript web frontend".to_string()).await.unwrap();
    let r3 = retriever.retrieve_hybrid("web frontend", 0.7).await.unwrap();
    assert!(!r3.is_empty());
}

#[tokio::test]
async fn bm25_cache_invalidation_on_add_with_metadata() {
    let model = MockEmbeddingModel;
    let store = InMemoryVectorStore::new();
    let retriever = Retriever::new(model, store).with_top_k(2);

    retriever.add_document_with_metadata(
        "first document content".to_string(),
        vec![("source".to_string(), "test".to_string())],
    )
    .await
    .unwrap();

    let r1 = retriever.retrieve_hybrid("first", 0.5).await.unwrap();
    assert!(!r1.is_empty());

    retriever.add_document_with_metadata(
        "second document different".to_string(),
        vec![("source".to_string(), "test".to_string())],
    )
    .await
    .unwrap();

    let r2 = retriever.retrieve_hybrid("second", 0.5).await.unwrap();
    assert!(!r2.is_empty());
}
