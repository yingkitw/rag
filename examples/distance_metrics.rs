use rag::{
    vector_store::{Document, InMemoryVectorStore, VectorStore},
    DistanceMetric,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distance Metrics Comparison ===\n");

    let documents = vec![
        Document::new("Document at [1, 0, 0]".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0]),
        Document::new("Document at [0, 1, 0]".to_string())
            .with_embedding(vec![0.0, 1.0, 0.0]),
        Document::new("Document at [1, 1, 0]".to_string())
            .with_embedding(vec![1.0, 1.0, 0.0]),
        Document::new("Document at [0.9, 0.1, 0]".to_string())
            .with_embedding(vec![0.9, 0.1, 0.0]),
    ];

    let metrics = vec![
        (DistanceMetric::Cosine, "Cosine Similarity"),
        (DistanceMetric::Euclidean, "Euclidean Distance (negated)"),
        (DistanceMetric::DotProduct, "Dot Product"),
        (DistanceMetric::Manhattan, "Manhattan Distance (negated)"),
    ];

    let query = vec![1.0, 0.0, 0.0];

    for (metric, name) in metrics {
        println!("--- {} ---", name);
        let store = InMemoryVectorStore::with_metric(metric);
        store.add_batch(documents.clone()).await?;

        let results = store.search(&query, 4).await?;
        for (i, result) in results.iter().enumerate() {
            println!(
                "  {}. Score: {:.4} - {}",
                i + 1,
                result.score,
                result.document.content
            );
        }
        println!();
    }

    println!("=== Key Observations ===");
    println!("- Cosine: Best for comparing direction (ignores magnitude)");
    println!("- Euclidean: Best for spatial data (considers magnitude)");
    println!("- Dot Product: Fast but assumes vectors are normalized");
    println!("- Manhattan: L1 distance, more robust to outliers than Euclidean");

    println!("\nDemo completed successfully!");
    Ok(())
}
