use rag::ingestion::{Source, WikiSource};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Wikipedia Ingestion Example ===\n");

    let titles = vec![
        "Rust (programming language)",
        "Retrieval-augmented generation",
        "Vector database",
    ];

    for title in titles {
        println!("Fetching: {}", title);
        let src = WikiSource::new(title);
        let docs = src.extract().await?;

        for doc in &docs {
            println!("  Title: {}", doc.source);
            println!(
                "  Content length: {} characters",
                doc.content.len()
            );
            println!(
                "  URL: {}\n",
                doc.metadata.get("url").unwrap_or(&"N/A".to_string())
            );

            // Print first 200 characters as preview
            let preview = if doc.content.len() > 200 {
                format!("{}...", &doc.content[..200])
            } else {
                doc.content.clone()
            };
            println!("  Preview: {}\n", preview.replace('\n', " "));
        }
    }

    println!("Demo completed successfully!");
    Ok(())
}
