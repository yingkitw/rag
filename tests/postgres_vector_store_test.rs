#![cfg(feature = "postgres")]

use rag::vector_store::{Document, MetadataFilter, VectorStore};
use rag::{DistanceMetric, PostgresVectorStore};

fn postgres_url() -> String {
    std::env::var("POSTGRES_TEST_URL")
        .or_else(|_| std::env::var("POSTGRES_URL"))
        .unwrap_or_default()
}

async fn connect_test_store(prefix: &str) -> Option<(PostgresVectorStore, String)> {
    let conn_str = postgres_url();
    let table = format!("{}_{}", prefix, uuid::Uuid::new_v4().to_string().replace('-', "_"));
    match PostgresVectorStore::connect(&conn_str, &table).await {
        Ok(store) => Some((store, table)),
        Err(e) => {
            eprintln!("Skipping Postgres test ({}): {}", table, e);
            None
        }
    }
}

#[tokio::test]
async fn test_postgres_vector_store_basic_operations() {
    let Some((store, _table)) = connect_test_store("test_basic").await else {
        return;
    };

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
    assert!(results[0].score > results[1].score);

    let filter = MetadataFilter::new().add("lang".to_string(), "python".to_string());
    let filtered = store
        .search_with_filter(&[1.0, 0.0, 0.0], 2, &filter)
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].document.content, "Python is easy");

    store.drop_table().await.unwrap();
}

#[tokio::test]
async fn test_postgres_vector_store_metrics() {
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let Some((store, _table)) = connect_test_store("test_metric").await else {
            return;
        };
        let store = store.with_metric(metric);

        let doc = Document::new("aligned".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        store.add(doc).await.unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "metric {:?} should return score ~1.0 for identical vectors",
            metric
        );

        store.drop_table().await.unwrap();
    }
}

#[tokio::test]
async fn test_postgres_vector_store_batch_and_clear() {
    let Some((store, _table)) = connect_test_store("test_batch").await else {
        return;
    };

    let docs = vec![
        Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]),
        Document::new("b".to_string()).with_embedding(vec![0.0, 1.0]),
        Document::new("c".to_string()).with_embedding(vec![0.5, 0.5]),
    ];
    store.add_batch(docs).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 3);

    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);

    store.drop_table().await.unwrap();
}
