use rag::vector_store::{Document, VectorStore};
use rag::PostgresVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn_str = std::env::var("POSTGRES_URL")
        .unwrap_or_else(|_| "host=localhost user=postgres password=postgres dbname=postgres".to_string());

    println!("=== PostgreSQL Vector Store Example ===\n");
    println!("Connecting to Postgres...");

    let store = PostgresVectorStore::connect(&conn_str, "rag_example_vectors").await?;

    let docs = vec![
        Document::new("Rust is a systems programming language".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0]),
        Document::new("Python is great for scripting".to_string())
            .with_embedding(vec![0.0, 1.0, 0.0]),
        Document::new("Go is built for concurrency".to_string())
            .with_embedding(vec![0.0, 0.0, 1.0]),
    ];

    store.add_batch(docs).await?;
    println!("Added 3 documents to Postgres via pgvector");

    let results = store.search(&[1.0, 0.0, 0.0], 2).await?;
    println!("\nTop 2 results for query [1.0, 0.0, 0.0]:");
    for (i, sim) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, sim.document.content, sim.score);
    }

    println!("\nDocument count: {}", store.count().await?);

    // Clean up the example table.
    store.drop_table().await?;
    println!("\nDropped example table.");

    Ok(())
}
