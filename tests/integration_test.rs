use rag::{
    chunker::{FixedSizeChunker, TextChunker},
    embeddings::OllamaEmbeddingModel,
    retriever::Retriever,
    vector_store::{Document, InMemoryVectorStore, cosine_similarity, VectorStore},
};

#[tokio::test]
async fn test_vector_store() {
    let store = InMemoryVectorStore::new();
    
    let doc = Document::new("test content".to_string())
        .with_embedding(vec![0.1, 0.2, 0.3]);
    let doc_id = doc.id.clone();
    
    store.add(doc).await.unwrap();
    
    let count = store.count().await.unwrap();
    assert_eq!(count, 1);
    
    let retrieved = store.get(&doc_id).await.unwrap();
    assert!(retrieved.is_some());
    
    let deleted = store.delete(&doc_id).await.unwrap();
    assert!(deleted);
    
    let count = store.count().await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_text_chunker() {
    let chunker = FixedSizeChunker::new(10, 2);
    
    let text = "one two three four five six seven eight nine ten eleven twelve thirteen";
    let chunks = chunker.chunk(text).unwrap();
    
    assert!(!chunks.is_empty());
    assert!(chunks.len() > 1);
}

#[tokio::test]
async fn test_document_creation() {
    let doc = Document::new("content".to_string())
        .with_metadata("key".to_string(), "value".to_string());
    
    assert_eq!(doc.content, "content");
    assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
}

#[tokio::test]
async fn test_vector_store_search() {
    let store = InMemoryVectorStore::new();
    
    let doc1 = Document::new("rust programming".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);
    let doc2 = Document::new("python programming".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0]);
    let doc3 = Document::new("rust development".to_string())
        .with_embedding(vec![0.9, 0.1, 0.0]);
    
    store.add(doc1).await.unwrap();
    store.add(doc2).await.unwrap();
    store.add(doc3).await.unwrap();
    
    let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
    
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score);
}

#[tokio::test]
async fn test_retriever_basic() {
    let embedding_model = OllamaEmbeddingModel::new("nomic-embed-text".to_string());
    let vector_store = InMemoryVectorStore::new();
    
    let retriever = Retriever::new(embedding_model, vector_store)
        .with_chunker(Box::new(FixedSizeChunker::new(20, 5)))
        .with_top_k(2);
    
    if retriever
        .add_document("Rust is a systems programming language".to_string())
        .await
        .is_err()
    {
        eprintln!("Skipping test_retriever_basic: Ollama not available");
        return;
    }
    
    let count = retriever.vector_store().count().await.unwrap();
    assert!(count > 0);
}

#[tokio::test]
async fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];
    
    let sim_ab = cosine_similarity(&a, &b);
    let sim_ac = cosine_similarity(&a, &c);
    
    assert!((sim_ab - 1.0).abs() < 0.001);
    assert!((sim_ac - 0.0).abs() < 0.001);
}