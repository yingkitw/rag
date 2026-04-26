use rag::{
    vector_store::{Document, InMemoryVectorStore, VectorStore},
    DistanceMetric,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pure Memory RAG (No Network Required) ===\n");

    let store = InMemoryVectorStore::with_metric(DistanceMetric::Cosine);

    let docs = vec![
        Document::new("Rust is a systems programming language with memory safety guarantees".to_string())
            .with_embedding(vec![0.95, 0.10, 0.05, 0.00])
            .with_metadata("topic".to_string(), "programming".to_string()),
        Document::new("Python is great for data science and machine learning".to_string())
            .with_embedding(vec![0.10, 0.90, 0.20, 0.05])
            .with_metadata("topic".to_string(), "programming".to_string()),
        Document::new("Machine learning uses neural networks to learn patterns".to_string())
            .with_embedding(vec![0.15, 0.20, 0.85, 0.10])
            .with_metadata("topic".to_string(), "ai".to_string()),
        Document::new("Deep learning is a subset of machine learning with deep neural networks".to_string())
            .with_embedding(vec![0.20, 0.25, 0.80, 0.15])
            .with_metadata("topic".to_string(), "ai".to_string()),
        Document::new("Docker containers package applications with their dependencies".to_string())
            .with_embedding(vec![0.30, 0.40, 0.10, 0.85])
            .with_metadata("topic".to_string(), "devops".to_string()),
    ];

    store.add_batch(docs).await?;
    println!("Added {} documents\n", store.count().await?);

    println!("=== Semantic Search ===");
    let rust_query = vec![0.9, 0.1, 0.0, 0.0];
    let rust_results = store.search(&rust_query, 2).await?;
    println!("Query: 'programming language'");
    for (i, r) in rust_results.iter().enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, r.score, r.document.content);
    }
    println!();

    let ai_query = vec![0.1, 0.1, 0.9, 0.1];
    let ai_results = store.search(&ai_query, 2).await?;
    println!("Query: 'machine learning'");
    for (i, r) in ai_results.iter().enumerate() {
        println!("  {}. [{:.3}] {}", i + 1, r.score, r.document.content);
    }
    println!();

    println!("=== Batch Query ===");
    let batch_queries = vec![
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.1, 0.1, 0.9, 0.1],
        vec![0.3, 0.3, 0.1, 0.8],
    ];
    let batch_results = store.search_batch(&batch_queries, 2).await?;
    for (qi, results) in batch_results.iter().enumerate() {
        println!("Batch query {}:", qi + 1);
        for (i, r) in results.iter().enumerate() {
            println!("  {}. [{:.3}] {}", i + 1, r.score, r.document.content);
        }
    }
    println!();

    println!("=== Document Lifecycle ===");
    let all_docs = store.list(10, 0).await?;
    println!("Total documents: {}", all_docs.len());

    if let Some(first) = all_docs.first() {
        println!("Deleting first document: {}", first.content);
        store.delete(&first.id).await?;
        println!("After delete: {} documents\n", store.count().await?);
    }

    store.clear().await?;
    println!("After clear: {} documents", store.count().await?);

    println!("\nDemo completed successfully!");
    Ok(())
}
