use rag::ingestion::{CodebaseSource, Source};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Codebase Ingestion Example ===\n");

    // Ingest the current project as a codebase
    let src = CodebaseSource::new(".")
        .with_extensions(vec![
            "rs".to_string(),
            "toml".to_string(),
            "md".to_string(),
        ])
        .with_max_file_size(1024 * 1024); // 1 MB per file

    let docs = src.extract().await?;

    println!("Extracted {} documents from codebase\n", docs.len());

    for (i, doc) in docs.iter().take(10).enumerate() {
        println!("{}. {} ({} bytes)", i + 1, doc.source, doc.content.len());
        if let Some(ext) = doc.metadata.get("extension") {
            println!("   Extension: {}", ext);
        }
    }

    if docs.len() > 10 {
        println!("\n... and {} more documents", docs.len() - 10);
    }

    println!("\n=== Summary ===");
    let total_bytes: usize = docs.iter().map(|d| d.content.len()).sum();
    println!("Total documents: {}", docs.len());
    println!("Total content size: {} bytes", total_bytes);

    println!("\nDemo completed successfully!");
    Ok(())
}
