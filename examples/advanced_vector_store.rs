use rag::{Document, InMemoryVectorStore, MetadataFilter, VectorStore};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vector Store with Metadata Filtering ===\n");

    let vector_store = InMemoryVectorStore::new();

    let doc1 = Document::new("Rust is a systems programming language".to_string())
        .with_metadata("category".to_string(), "programming".to_string())
        .with_metadata("language".to_string(), "Rust".to_string())
        .with_embedding(vec![1.0, 0.0, 0.0]);

    let doc2 = Document::new("Python is a high-level programming language".to_string())
        .with_metadata("category".to_string(), "programming".to_string())
        .with_metadata("language".to_string(), "Python".to_string())
        .with_embedding(vec![0.0, 1.0, 0.0]);

    let doc3 = Document::new("Machine learning is a subset of AI".to_string())
        .with_metadata("category".to_string(), "AI".to_string())
        .with_metadata("language".to_string(), "English".to_string())
        .with_embedding(vec![0.0, 0.0, 1.0]);

    let doc4 = Document::new("Deep learning uses neural networks".to_string())
        .with_metadata("category".to_string(), "AI".to_string())
        .with_metadata("language".to_string(), "English".to_string())
        .with_embedding(vec![0.5, 0.5, 0.5]);

    vector_store.add(doc1).await?;
    vector_store.add(doc2).await?;
    vector_store.add(doc3).await?;
    vector_store.add(doc4).await?;

    println!("Added 4 documents to the vector store\n");

    let count = vector_store.count().await?;
    println!("Total documents: {}\n", count);

    println!("=== All Documents ===");
    let all_docs = vector_store.list(10, 0).await?;
    for (i, doc) in all_docs.iter().enumerate() {
        println!("{}. {} ({})", i + 1, doc.content, doc.metadata.get("category").unwrap_or(&"?".to_string()));
    }
    println!();

    let query = vec![0.2, 0.2, 0.8];

    println!("=== Search without filter ===");
    let results = vector_store.search(&query, 3).await?;
    for (i, result) in results.iter().enumerate() {
        println!("{}. {:.3} - {}", i + 1, result.score, result.document.content);
    }
    println!();

    println!("=== Search with category filter (AI) ===");
    let filter = MetadataFilter::new().add("category".to_string(), "AI".to_string());
    let results = vector_store.search_with_filter(&query, 3, &filter).await?;
    for (i, result) in results.iter().enumerate() {
        println!("{}. {:.3} - {}", i + 1, result.score, result.document.content);
    }
    println!();

    println!("=== Search with language filter (Python) ===");
    let filter = MetadataFilter::new().add("language".to_string(), "Python".to_string());
    let results = vector_store.search_with_filter(&query, 3, &filter).await?;
    for (i, result) in results.iter().enumerate() {
        println!("{}. {:.3} - {}", i + 1, result.score, result.document.content);
    }
    println!();

    println!("=== Batch operations ===");
    let batch_docs = vec![
        Document::new("Rust's ownership system ensures memory safety".to_string())
            .with_metadata("category".to_string(), "programming".to_string())
            .with_metadata("language".to_string(), "Rust".to_string())
            .with_embedding(vec![0.8, 0.1, 0.1]),
        Document::new("Python's dynamic typing makes it easy to learn".to_string())
            .with_metadata("category".to_string(), "programming".to_string())
            .with_metadata("language".to_string(), "Python".to_string())
            .with_embedding(vec![0.1, 0.8, 0.1]),
    ];

    vector_store.add_batch(batch_docs).await?;
    let count = vector_store.count().await?;
    println!("After batch add, total documents: {}\n", count);

    println!("=== Persistence (Save to file) ===");
    vector_store.save_to_file("vector_store_backup.json").await?;
    println!("Saved vector store to vector_store_backup.json\n");

    println!("=== Load from file ===");
    let loaded_store = InMemoryVectorStore::load_from_file("vector_store_backup.json").await?;
    let loaded_count = loaded_store.count().await?;
    println!("Loaded {} documents from file\n", loaded_count);

    println!("=== Delete document ===");
    let all_docs = vector_store.list(10, 0).await?;
    if let Some(first_doc) = all_docs.first() {
        let deleted = vector_store.delete(&first_doc.id).await?;
        println!("Deleted document with ID: {} - {}", deleted, first_doc.id);
        let count = vector_store.count().await?;
        println!("Remaining documents: {}\n", count);
    }

    println!("=== Batch delete ===");
    let all_docs = vector_store.list(10, 0).await?;
    let ids_to_delete: Vec<String> = all_docs.iter().take(2).map(|d| d.id.clone()).collect();
    let deleted_count = vector_store.delete_batch(ids_to_delete).await?;
    println!("Deleted {} documents", deleted_count);
    let count = vector_store.count().await?;
    println!("Remaining documents: {}\n", count);

    println!("=== Clear all ===");
    vector_store.clear().await?;
    let count = vector_store.count().await?;
    println!("After clear, total documents: {}\n", count);

    std::fs::remove_file("vector_store_backup.json")?;

    println!("Demo completed successfully!");
    Ok(())
}
