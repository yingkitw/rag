use rag::{
    vector_store::{Document, InMemoryVectorStore, MetadataFilter, MinimalVectorDB, VectorStore},
    DistanceMetric, Index, FlatIndex,
};

// ============================================================
// Tests validating patterns from examples/batch_search.rs
// ============================================================

#[tokio::test]
async fn test_batch_search_ranking_consistency() {
    let store = InMemoryVectorStore::new();

    let docs = vec![
        Document::new("Rust".to_string()).with_embedding(vec![1.0, 0.0, 0.0, 0.0]),
        Document::new("Go".to_string()).with_embedding(vec![0.9, 0.1, 0.0, 0.0]),
        Document::new("Python".to_string()).with_embedding(vec![0.0, 1.0, 0.0, 0.0]),
        Document::new("JavaScript".to_string()).with_embedding(vec![0.0, 0.0, 1.0, 0.0]),
        Document::new("TypeScript".to_string()).with_embedding(vec![0.0, 0.1, 0.9, 0.0]),
    ];
    store.add_batch(docs).await.unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.0, 0.0],
    ];

    let results = store.search_batch(&queries, 2).await.unwrap();
    assert_eq!(results.len(), 3);

    // Query 1: should find Rust and Go
    assert_eq!(results[0].len(), 2);
    assert_eq!(results[0][0].document.content, "Rust");
    assert_eq!(results[0][1].document.content, "Go");
    assert!(results[0][0].score >= results[0][1].score);

    // Query 2: should find JavaScript and TypeScript
    assert_eq!(results[1].len(), 2);
    assert_eq!(results[1][0].document.content, "JavaScript");
    assert_eq!(results[1][1].document.content, "TypeScript");

    // Query 3: should find Rust and Python (mixed)
    assert_eq!(results[2].len(), 2);
    assert!(results[2][0].score >= results[2][1].score);
}

#[tokio::test]
async fn test_batch_search_euclidean_spatial() {
    let store = InMemoryVectorStore::with_metric(DistanceMetric::Euclidean);

    let docs = vec![
        Document::new("Point A".to_string()).with_embedding(vec![0.0, 0.0, 0.0]),
        Document::new("Point B".to_string()).with_embedding(vec![0.1, 0.1, 0.1]),
        Document::new("Point C".to_string()).with_embedding(vec![10.0, 10.0, 10.0]),
    ];
    store.add_batch(docs).await.unwrap();

    let queries = vec![vec![0.0, 0.0, 0.0], vec![5.0, 5.0, 5.0]];
    let results = store.search_batch(&queries, 2).await.unwrap();

    // Query at origin should find Point A closest, then Point B
    assert_eq!(results[0][0].document.content, "Point A");
    assert_eq!(results[0][1].document.content, "Point B");

    // Query at [5,5,5] should find Point C (at [10,10,10], dist ~8.66) vs Point B (dist ~8.48)
    // Actually Point B is closer: sqrt((5-0.1)^2*3) ~ 8.48, Point C: sqrt(5^2*3) ~ 8.66
    assert_eq!(results[1][0].document.content, "Point B");
}

// ============================================================
// Tests validating patterns from examples/distance_metrics.rs
// ============================================================

#[tokio::test]
async fn test_distance_metrics_produce_different_rankings() {
    let docs = vec![
        Document::new("A".to_string()).with_embedding(vec![1.0, 0.0, 0.0]),
        Document::new("B".to_string()).with_embedding(vec![0.0, 1.0, 0.0]),
        Document::new("C".to_string()).with_embedding(vec![1.0, 1.0, 0.0]),
        Document::new("D".to_string()).with_embedding(vec![0.9, 0.1, 0.0]),
    ];
    let query = vec![1.0, 0.0, 0.0];

    // Cosine should rank D very close to A (similar direction)
    let cosine_store = InMemoryVectorStore::with_metric(DistanceMetric::Cosine);
    cosine_store.add_batch(docs.clone()).await.unwrap();
    let cosine_results = cosine_store.search(&query, 4).await.unwrap();
    assert_eq!(cosine_results[0].document.content, "A");
    // D has similar direction, should be 2nd
    assert_eq!(cosine_results[1].document.content, "D");

    // Dot product: A=[1,0,0] dot [1,0,0]=1, D=[0.9,0.1,0] dot [1,0,0]=0.9, C=1, B=0
    // A=1, C=1 (tie -- depends on DashMap iteration order), D=0.9, B=0
    let dot_store = InMemoryVectorStore::with_metric(DistanceMetric::DotProduct);
    dot_store.add_batch(docs.clone()).await.unwrap();
    let dot_results = dot_store.search(&query, 4).await.unwrap();
    // First result should be either A or C (both score 1.0), we just check it's not B
    assert!(dot_results[0].score > 0.99);
    assert_ne!(dot_results[3].document.content, "A");
    assert_ne!(dot_results[3].document.content, "C");

    // Euclidean: A at distance 0, D at distance 0.1, C at distance 1, B at distance sqrt(2)
    let euclid_store = InMemoryVectorStore::with_metric(DistanceMetric::Euclidean);
    euclid_store.add_batch(docs.clone()).await.unwrap();
    let euclid_results = euclid_store.search(&query, 4).await.unwrap();
    assert_eq!(euclid_results[0].document.content, "A");
    assert_eq!(euclid_results[1].document.content, "D");
    assert_eq!(euclid_results[2].document.content, "C");
}

#[tokio::test]
async fn test_cosine_ignores_magnitude() {
    let store = InMemoryVectorStore::with_metric(DistanceMetric::Cosine);

    let docs = vec![
        Document::new("Small".to_string()).with_embedding(vec![1.0, 0.0]),
        Document::new("Large".to_string()).with_embedding(vec![100.0, 0.0]),
        Document::new("Orthogonal".to_string()).with_embedding(vec![0.0, 1.0]),
    ];
    store.add_batch(docs).await.unwrap();

    let results = store.search(&[1.0, 0.0], 3).await.unwrap();
    // Cosine should see Small and Large as equally similar (same direction)
    assert!(results[0].score > 0.99); // Small
    assert!(results[1].score > 0.99); // Large (same direction)
    assert!(results[2].score < 0.01); // Orthogonal
}

// ============================================================
// Tests validating patterns from examples/pure_memory_rag.rs
// ============================================================

#[tokio::test]
async fn test_pure_memory_rag_document_lifecycle() {
    let store = InMemoryVectorStore::with_metric(DistanceMetric::Cosine);

    let docs = vec![
        Document::new("Rust doc".to_string())
            .with_embedding(vec![0.95, 0.10, 0.05, 0.00])
            .with_metadata("topic".to_string(), "programming".to_string()),
        Document::new("Python doc".to_string())
            .with_embedding(vec![0.10, 0.90, 0.20, 0.05])
            .with_metadata("topic".to_string(), "programming".to_string()),
        Document::new("ML doc".to_string())
            .with_embedding(vec![0.15, 0.20, 0.85, 0.10])
            .with_metadata("topic".to_string(), "ai".to_string()),
        Document::new("Deep learning doc".to_string())
            .with_embedding(vec![0.20, 0.25, 0.80, 0.15])
            .with_metadata("topic".to_string(), "ai".to_string()),
        Document::new("Docker doc".to_string())
            .with_embedding(vec![0.30, 0.40, 0.10, 0.85])
            .with_metadata("topic".to_string(), "devops".to_string()),
    ];

    store.add_batch(docs).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 5);

    // Semantic search for programming
    let rust_results = store.search(&[0.9, 0.1, 0.0, 0.0], 2).await.unwrap();
    assert_eq!(rust_results[0].document.content, "Rust doc");

    // Semantic search for AI
    let ai_results = store.search(&[0.1, 0.1, 0.9, 0.1], 2).await.unwrap();
    assert_eq!(ai_results[0].document.content, "ML doc");

    // Batch query
    let batch_queries = vec![
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.1, 0.1, 0.9, 0.1],
        vec![0.3, 0.3, 0.1, 0.8],
    ];
    let batch_results = store.search_batch(&batch_queries, 2).await.unwrap();
    assert_eq!(batch_results.len(), 3);
    assert_eq!(batch_results[0][0].document.content, "Rust doc");
    assert_eq!(batch_results[1][0].document.content, "ML doc");
    assert_eq!(batch_results[2][0].document.content, "Docker doc");

    // Document lifecycle
    let all_docs = store.list(10, 0).await.unwrap();
    assert_eq!(all_docs.len(), 5);

    store.delete(&all_docs[0].id).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 4);

    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_metadata_filter_with_embedding_search() {
    let store = InMemoryVectorStore::new();

    let docs = vec![
        Document::new("Rust programming".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0])
            .with_metadata("lang".to_string(), "rust".to_string()),
        Document::new("Rust web framework".to_string())
            .with_embedding(vec![0.9, 0.1, 0.0])
            .with_metadata("lang".to_string(), "rust".to_string()),
        Document::new("Python data science".to_string())
            .with_embedding(vec![0.0, 1.0, 0.0])
            .with_metadata("lang".to_string(), "python".to_string()),
    ];
    store.add_batch(docs).await.unwrap();

    // Search all returns Rust docs first
    let all_results = store.search(&[1.0, 0.0, 0.0], 3).await.unwrap();
    assert_eq!(all_results.len(), 3);

    // Filtered search only returns rust docs
    let filter = MetadataFilter::new().add("lang".to_string(), "rust".to_string());
    let filtered_results = store.search_with_filter(&[1.0, 0.0, 0.0], 3, &filter).await.unwrap();
    assert_eq!(filtered_results.len(), 2);
    assert!(filtered_results.iter().all(|r| r.document.metadata.get("lang") == Some(&"rust".to_string())));
}

// ============================================================
// Tests validating MinimalVectorDB equivalency
// ============================================================

#[tokio::test]
async fn test_minimal_db_equivalent_behavior() {
    let inmem = InMemoryVectorStore::new();
    let minimal = MinimalVectorDB::new();

    let docs = vec![
        Document::new("Doc 1".to_string()).with_embedding(vec![1.0, 0.0]),
        Document::new("Doc 2".to_string()).with_embedding(vec![0.0, 1.0]),
    ];

    inmem.add_batch(docs.clone()).await.unwrap();
    minimal.add_batch(docs.clone()).await.unwrap();

    let inmem_results = inmem.search(&[1.0, 0.0], 2).await.unwrap();
    let minimal_results = minimal.search(&[1.0, 0.0], 2).await.unwrap();

    assert_eq!(inmem_results.len(), minimal_results.len());
    assert_eq!(inmem_results[0].document.content, minimal_results[0].document.content);
    assert_eq!(inmem_results[1].document.content, minimal_results[1].document.content);

    assert_eq!(inmem.count().await.unwrap(), minimal.count().await.unwrap());

    inmem.clear().await.unwrap();
    minimal.clear().await.unwrap();
    assert_eq!(inmem.count().await.unwrap(), 0);
    assert_eq!(minimal.count().await.unwrap(), 0);
}

// ============================================================
// Tests validating FlatIndex directly (from Index trait)
// ============================================================

#[test]
fn test_flat_index_metric_accessor() {
    let index_cosine = FlatIndex::new();
    assert_eq!(index_cosine.metric(), DistanceMetric::Cosine);

    let index_euclid = FlatIndex::with_metric(DistanceMetric::Euclidean);
    assert_eq!(index_euclid.metric(), DistanceMetric::Euclidean);
}

#[test]
fn test_flat_index_batch_search_parallel() {
    let index = FlatIndex::new();
    index.add(Document::new("A".to_string()).with_embedding(vec![1.0, 0.0]));
    index.add(Document::new("B".to_string()).with_embedding(vec![0.0, 1.0]));
    index.add(Document::new("C".to_string()).with_embedding(vec![0.5, 0.5]));

    let queries = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
    ];

    let results = index.search_batch(&queries, 2);
    assert_eq!(results.len(), 3);

    // Each query should return 2 results
    for r in &results {
        assert_eq!(r.len(), 2);
    }

    // Query 1 should find A first
    assert_eq!(results[0][0].document.content, "A");
    // Query 2 should find B first
    assert_eq!(results[1][0].document.content, "B");
    // Query 3 should find C first (perfect match)
    assert_eq!(results[2][0].document.content, "C");
}
