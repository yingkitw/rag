use rag::ingestion::{PdfSource, Source};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PDF Ingestion Example ===\n");

    let pdf_path = env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --example ingest_pdf -- <path/to/file.pdf>");
            std::process::exit(1);
        });

    println!("Loading PDF: {}\n", pdf_path);

    let src = PdfSource::new(&pdf_path);
    let docs = src.extract().await?;

    for doc in &docs {
        println!("Source: {}", doc.source);
        println!("Format: {}", doc.metadata.get("format").unwrap_or(&"unknown".to_string()));
        println!("Path: {}", doc.metadata.get("path").unwrap_or(&"unknown".to_string()));
        println!(
            "Content length: {} characters",
            doc.content.len()
        );
        println!("\n--- Content Preview (first 500 chars) ---");
        let preview = if doc.content.len() > 500 {
            format!("{}...", &doc.content[..500])
        } else {
            doc.content.clone()
        };
        println!("{}\n", preview);
    }

    println!("Demo completed successfully!");
    Ok(())
}
