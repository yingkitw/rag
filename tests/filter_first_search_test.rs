use rag::vector_store::{Document, InMemoryVectorStore, MetadataFilter, VectorStore};

#[tokio::test]
async fn filter_first_search_exact_match() {
    let store = InMemoryVectorStore::new();

    let mut d1 = Document::new("rust programming".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
    d1.metadata.insert("lang".to_string(), "rust".to_string());

    let mut d2 = Document::new("python scripting".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);
    d2.metadata.insert("lang".to_string(), "python".to_string());

    let mut d3 = Document::new("rust cli tools".to_string()).with_embedding(vec![0.9, 0.1, 0.0]);
    d3.metadata.insert("lang".to_string(), "rust".to_string());

    store.add(d1).await.unwrap();
    store.add(d2).await.unwrap();
    store.add(d3).await.unwrap();

    let filter = MetadataFilter::new().add("lang".to_string(), "rust".to_string());
    let results = store.search_with_filter(&[1.0, 0.0, 0.0], 5, &filter).await.unwrap();

    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r.document.metadata.get("lang"), Some(&"rust".to_string()));
    }
    // Should be ordered by similarity
    assert!(results[0].score >= results[1].score);
}

#[tokio::test]
async fn filter_first_no_match() {
    let store = InMemoryVectorStore::new();
    let mut d = Document::new("content".to_string()).with_embedding(vec![1.0, 0.0]);
    d.metadata.insert("tag".to_string(), "A".to_string());
    store.add(d).await.unwrap();

    let filter = MetadataFilter::new().add("tag".to_string(), "B".to_string());
    let results = store.search_with_filter(&[1.0, 0.0], 5, &filter).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn filter_first_empty_filter_returns_all() {
    let store = InMemoryVectorStore::new();
    let d = Document::new("content".to_string()).with_embedding(vec![1.0, 0.0]);
    store.add(d).await.unwrap();

    let filter = MetadataFilter::new();
    let results = store.search_with_filter(&[1.0, 0.0], 5, &filter).await.unwrap();
    assert_eq!(results.len(), 1);
}
