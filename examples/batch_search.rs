use rag::{
    vector_store::{Document, InMemoryVectorStore, VectorStore},
    DistanceMetric,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Search Example ===\n");

    let store = InMemoryVectorStore::new();

    let docs = vec![
        Document::new("Rust programming language".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0, 0.0]),
        Document::new("Python data science".to_string())
            .with_embedding(vec![0.0, 1.0, 0.0, 0.0]),
        Document::new("JavaScript web development".to_string())
            .with_embedding(vec![0.0, 0.0, 1.0, 0.0]),
        Document::new("Go systems programming".to_string())
            .with_embedding(vec![0.9, 0.1, 0.0, 0.0]),
        Document::new("TypeScript typed JavaScript".to_string())
            .with_embedding(vec![0.0, 0.1, 0.9, 0.0]),
    ];

    store.add_batch(docs).await?;

    println!("Added 5 documents to the vector store\n");

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.0, 0.0],
    ];

    println!("Running batch search for {} queries...\n", queries.len());

    let results = store.search_batch(&queries, 2).await?;

    for (i, query_results) in results.iter().enumerate() {
        println!("Query {} top results:", i + 1);
        for (j, similarity) in query_results.iter().enumerate() {
            println!(
                "  {}. Score: {:.4} - {}",
                j + 1,
                similarity.score,
                similarity.document.content
            );
        }
        println!();
    }

    println!("=== Batch Search with Euclidean Metric ===\n");

    let euclidean_store = InMemoryVectorStore::with_metric(DistanceMetric::Euclidean);

    let docs = vec![
        Document::new("Point A at origin".to_string())
            .with_embedding(vec![0.0, 0.0, 0.0]),
        Document::new("Point B near origin".to_string())
            .with_embedding(vec![0.1, 0.1, 0.1]),
        Document::new("Point C far away".to_string())
            .with_embedding(vec![10.0, 10.0, 10.0]),
    ];

    euclidean_store.add_batch(docs).await?;

    let queries = vec![vec![0.0, 0.0, 0.0], vec![5.0, 5.0, 5.0]];
    let results = euclidean_store.search_batch(&queries, 2).await?;

    for (i, query_results) in results.iter().enumerate() {
        println!("Euclidean Query {}:", i + 1);
        for (j, similarity) in query_results.iter().enumerate() {
            println!(
                "  {}. Score: {:.4} - {}",
                j + 1,
                similarity.score,
                similarity.document.content
            );
        }
        println!();
    }

    println!("Demo completed successfully!");
    Ok(())
}
