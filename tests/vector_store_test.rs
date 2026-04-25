use rag::{Document, InMemoryVectorStore, MetadataFilter, MinimalVectorDB, VectorStore};
use std::fs;

#[tokio::test]
async fn test_vector_store_basic_operations() {
    let store = InMemoryVectorStore::new();

    let doc1 = Document::new("Test document 1".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    let doc2 = Document::new("Test document 2".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0]);

    store.add(doc1.clone()).await.unwrap();
    store.add(doc2.clone()).await.unwrap();

    assert_eq!(store.count().await.unwrap(), 2);

    let retrieved = store.get(&doc1.id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "Test document 1");

    let list = store.list(10, 0).await.unwrap();
    assert_eq!(list.len(), 2);

    let deleted = store.delete(&doc1.id).await.unwrap();
    assert!(deleted);

    assert_eq!(store.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_vector_store_batch_operations() {
    let store = MinimalVectorDB::new();

    let docs: Vec<Document> = (1..=5)
        .map(|i| {
            Document::new(format!("Batch document {}", i))
                .with_embedding(vec![i as f32 / 5.0, 0.0, 0.0])
        })
        .collect();

    store.add_batch(docs.clone()).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 5);

    let ids: Vec<String> = docs.iter().take(3).map(|d| d.id.clone()).collect();
    let deleted = store.delete_batch(ids).await.unwrap();
    assert_eq!(deleted, 3);

    assert_eq!(store.count().await.unwrap(), 2);

    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_vector_store_search() {
    let store = InMemoryVectorStore::new();

    let doc1 = Document::new("Similar document".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    let doc2 = Document::new("Dissimilar document".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0]);

    let doc3 = Document::new("Somewhat similar".to_string())
        .with_embedding(vec![0.7, 0.3, 0.0]);

    store.add(doc1).await.unwrap();
    store.add(doc2).await.unwrap();
    store.add(doc3).await.unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 2).await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].score > results[1].score);
    assert!(results[0].score > 0.9);
}

#[tokio::test]
async fn test_vector_store_search_without_embedding() {
    let store = MinimalVectorDB::new();

    let doc1 = Document::new("Document without embedding".to_string());
    let doc2 = Document::new("Document with embedding".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    store.add(doc1).await.unwrap();
    store.add(doc2).await.unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 10).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.content, "Document with embedding");
}

#[tokio::test]
async fn test_vector_store_metadata_filter() {
    let store = InMemoryVectorStore::new();

    let doc1 = Document::new("Rust programming".to_string())
        .with_metadata("category".to_string(), "programming".to_string())
        .with_metadata("language".to_string(), "Rust".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    let doc2 = Document::new("Python programming".to_string())
        .with_metadata("category".to_string(), "programming".to_string())
        .with_metadata("language".to_string(), "Python".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0]);

    let doc3 = Document::new("AI concepts".to_string())
        .with_metadata("category".to_string(), "AI".to_string())
        .with_embedding(vec![0.0, 0.0, 1.0]);

    store.add(doc1).await.unwrap();
    store.add(doc2).await.unwrap();
    store.add(doc3).await.unwrap();

    let query = vec![0.5, 0.5, 0.5];

    let filter = MetadataFilter::new().add("category".to_string(), "programming".to_string());
    let results = store.search_with_filter(&query, 10, &filter).await.unwrap();
    assert_eq!(results.len(), 2);

    let filter = MetadataFilter::new().add("language".to_string(), "Rust".to_string());
    let results = store.search_with_filter(&query, 10, &filter).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.content, "Rust programming");

    let filter = MetadataFilter::new()
        .add("category".to_string(), "programming".to_string())
        .add("language".to_string(), "Python".to_string());
    let results = store.search_with_filter(&query, 10, &filter).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.content, "Python programming");
}

#[tokio::test]
async fn test_vector_store_empty_filter() {
    let store = MinimalVectorDB::new();

    let doc = Document::new("Test document".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    store.add(doc).await.unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let filter = MetadataFilter::new();
    let results = store.search_with_filter(&query, 10, &filter).await.unwrap();

    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_vector_store_filter_nonexistent_metadata() {
    let store = InMemoryVectorStore::new();

    let doc = Document::new("Test document".to_string())
        .with_metadata("category".to_string(), "AI".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    store.add(doc).await.unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let filter = MetadataFilter::new().add("language".to_string(), "Rust".to_string());
    let results = store.search_with_filter(&query, 10, &filter).await.unwrap();

    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_vector_store_persistence_in_memory() {
    let filename = "test_vector_store_in_memory.json";
    let _guard = CleanupGuard::new(filename);

    let store = InMemoryVectorStore::new();

    let docs: Vec<Document> = (1..=3)
        .map(|i| {
            Document::new(format!("Persistent document {}", i))
                .with_metadata("id".to_string(), format!("doc_{}", i))
                .with_embedding(vec![i as f32 / 3.0, 0.0, 0.0])
        })
        .collect();

    store.add_batch(docs).await.unwrap();
    store.save_to_file(filename).await.unwrap();

    assert!(fs::metadata(filename).is_ok());

    let loaded_store = InMemoryVectorStore::load_from_file(filename).await.unwrap();
    assert_eq!(loaded_store.count().await.unwrap(), 3);

    let list = loaded_store.list(10, 0).await.unwrap();
    assert_eq!(list.len(), 3);
    assert!(list.iter().all(|doc| doc.content.starts_with("Persistent document")));
}

#[tokio::test]
async fn test_vector_store_persistence_minimal_db() {
    let filename = "test_vector_store_minimal.json";
    let _guard = CleanupGuard::new(filename);

    let store = MinimalVectorDB::new();

    let doc1 = Document::new("Minimal DB document".to_string())
        .with_metadata("type".to_string(), "test".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    store.add(doc1).await.unwrap();
    store.save_to_file(filename).await.unwrap();

    let loaded_store = MinimalVectorDB::load_from_file(filename).await.unwrap();
    assert_eq!(loaded_store.count().await.unwrap(), 1);

    let list = loaded_store.list(10, 0).await.unwrap();
    assert_eq!(list[0].content, "Minimal DB document");
    assert_eq!(list[0].metadata.get("type"), Some(&"test".to_string()));
}

#[tokio::test]
async fn test_vector_store_load_empty_file() {
    let filename = "test_empty_vector_store.json";
    let _guard = CleanupGuard::new(filename);

    fs::write(filename, "[]").unwrap();

    let store = InMemoryVectorStore::load_from_file(filename).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_vector_store_pagination() {
    let store = MinimalVectorDB::new();

    let docs: Vec<Document> = (1..=10)
        .map(|i| {
            Document::new(format!("Document {}", i))
                .with_embedding(vec![i as f32 / 10.0, 0.0, 0.0])
        })
        .collect();

    store.add_batch(docs).await.unwrap();

    let page1 = store.list(3, 0).await.unwrap();
    assert_eq!(page1.len(), 3);

    let page2 = store.list(3, 3).await.unwrap();
    assert_eq!(page2.len(), 3);

    let page3 = store.list(3, 6).await.unwrap();
    assert_eq!(page3.len(), 3);

    let page4 = store.list(3, 9).await.unwrap();
    assert_eq!(page4.len(), 1);
}

#[tokio::test]
async fn test_vector_store_top_k_limited() {
    let store = InMemoryVectorStore::new();

    let docs: Vec<Document> = (1..=10)
        .map(|i| {
            Document::new(format!("Document {}", i))
                .with_embedding(vec![i as f32 / 10.0, 0.0, 0.0])
        })
        .collect();

    store.add_batch(docs).await.unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 5).await.unwrap();

    assert_eq!(results.len(), 5);
    assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
}

#[tokio::test]
async fn test_vector_store_delete_nonexistent() {
    let store = MinimalVectorDB::new();

    let doc = Document::new("Test".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
    store.add(doc).await.unwrap();

    let deleted = store.delete("nonexistent-id").await.unwrap();
    assert!(!deleted);

    assert_eq!(store.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_vector_store_batch_delete_nonexistent() {
    let store = InMemoryVectorStore::new();

    let docs: Vec<Document> = (1..=3)
        .map(|i| Document::new(format!("Doc {}", i)).with_embedding(vec![i as f32 / 3.0, 0.0, 0.0]))
        .collect();

    store.add_batch(docs).await.unwrap();

    let ids = vec!["nonexistent1".to_string(), "nonexistent2".to_string()];
    let deleted = store.delete_batch(ids).await.unwrap();
    assert_eq!(deleted, 0);

    assert_eq!(store.count().await.unwrap(), 3);
}

#[tokio::test]
async fn test_vector_store_with_capacity() {
    let store = InMemoryVectorStore::with_capacity(100);
    let store2 = MinimalVectorDB::with_capacity(100);

    let doc = Document::new("Test".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);

    store.add(doc.clone()).await.unwrap();
    store2.add(doc).await.unwrap();

    assert_eq!(store.count().await.unwrap(), 1);
    assert_eq!(store2.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_document_creation() {
    let doc1 = Document::new("Content".to_string());
    assert!(!doc1.id.is_empty());
    assert_eq!(doc1.content, "Content");
    assert!(doc1.metadata.is_empty());
    assert!(doc1.embedding.is_none());

    let doc2 = Document::with_id("custom-id".to_string(), "Content 2".to_string());
    assert_eq!(doc2.id, "custom-id");
    assert_eq!(doc2.content, "Content 2");

    let doc3 = Document::new("Content 3".to_string())
        .with_metadata("key1".to_string(), "value1".to_string())
        .with_metadata("key2".to_string(), "value2".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    assert_eq!(doc3.metadata.len(), 2);
    assert_eq!(doc3.metadata.get("key1"), Some(&"value1".to_string()));
    assert!(doc3.embedding.is_some());
    assert_eq!(doc3.embedding.unwrap(), vec![1.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_cosine_similarity() {
    use rag::vector_store::cosine_similarity;

    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);

    let c = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &c);
    assert!((sim - 0.0).abs() < 1e-6);

    let d = vec![0.707, 0.707, 0.0];
    let sim = cosine_similarity(&a, &d);
    assert!((sim - 0.707).abs() < 0.01);

    let e = vec![];
    let f = vec![1.0];
    let sim = cosine_similarity(&e, &f);
    assert_eq!(sim, 0.0);
}

#[tokio::test]
async fn test_vector_store_search_empty() {
    let store = MinimalVectorDB::new();

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 10).await.unwrap();

    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_vector_store_large_scale() {
    let store = InMemoryVectorStore::new();

    let docs: Vec<Document> = (1..=1000)
        .map(|i| {
            let angle = (i as f32 / 1000.0) * std::f32::consts::PI / 4.0;
            Document::new(format!("Document {}", i))
                .with_embedding(vec![angle.cos(), angle.sin(), 0.0])
        })
        .collect();

    store.add_batch(docs).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1000);

    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 10).await.unwrap();

    assert_eq!(results.len(), 10);
    assert!(results[0].score >= results[9].score);
    assert!(results[0].score > 0.9);
}

struct CleanupGuard(String);

impl CleanupGuard {
    fn new(filename: &str) -> Self {
        Self(filename.to_string())
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
