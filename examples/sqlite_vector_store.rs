use rag::vector_store::{Document, VectorStore};
use rag::SqliteVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SQLite Vector Store Example ===\n");

    let store = SqliteVectorStore::open_in_memory()?;

    let docs = vec![
        Document::new("Rust is a systems programming language".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0]),
        Document::new("Python is great for scripting".to_string())
            .with_embedding(vec![0.0, 1.0, 0.0]),
        Document::new("Go is built for concurrency".to_string())
            .with_embedding(vec![0.0, 0.0, 1.0]),
    ];

    store.add_batch(docs).await?;
    println!("Added 3 documents to SQLite");

    let results = store.search(&[1.0, 0.0, 0.0], 2).await?;
    println!("\nTop 2 results for query [1.0, 0.0, 0.0]:");
    for (i, sim) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, sim.document.content, sim.score);
    }

    println!("\nDocument count: {}", store.count().await?);

    Ok(())
}
