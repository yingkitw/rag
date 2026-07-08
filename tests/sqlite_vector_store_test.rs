#![cfg(feature = "sqlite")]

use rag::SqliteVectorStore;
use rag::vector_store::{Document, MetadataFilter, VectorStore};

#[tokio::test]
async fn test_sqlite_vector_store_basic_operations() {
    let store = SqliteVectorStore::open_in_memory().unwrap();

    let doc1 = Document::new("Rust is fast".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0])
        .with_metadata("lang".to_string(), "rust".to_string());
    let doc2 = Document::new("Python is easy".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0])
        .with_metadata("lang".to_string(), "python".to_string());

    store.add(doc1.clone()).await.unwrap();
    store.add(doc2.clone()).await.unwrap();

    assert_eq!(store.count().await.unwrap(), 2);

    let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].document.content, "Rust is fast");

    let filter = MetadataFilter::new().add("lang".to_string(), "python".to_string());
    let filtered = store
        .search_with_filter(&[1.0, 0.0, 0.0], 2, &filter)
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].document.content, "Python is easy");
}

#[tokio::test]
async fn test_sqlite_vector_store_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");

    let doc_id = {
        let store = SqliteVectorStore::open(&path).unwrap();
        let doc = Document::new("persistent".to_string()).with_embedding(vec![0.5, 0.5]);
        let id = doc.id.clone();
        store.add(doc).await.unwrap();
        id
    };

    let store2 = SqliteVectorStore::open(&path).unwrap();
    let fetched = store2.get(&doc_id).await.unwrap();
    assert!(fetched.is_some());
    assert_eq!(fetched.unwrap().content, "persistent");
    assert_eq!(store2.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_sqlite_vector_store_batch_and_clear() {
    let store = SqliteVectorStore::open_in_memory().unwrap();

    let docs = vec![
        Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]),
        Document::new("b".to_string()).with_embedding(vec![0.0, 1.0]),
        Document::new("c".to_string()).with_embedding(vec![0.5, 0.5]),
    ];
    store.add_batch(docs).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 3);

    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}
