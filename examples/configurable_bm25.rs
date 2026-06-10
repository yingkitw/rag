//! Example: Configurable BM25 parameters and field-level boosts.
//!
//! Shows how k1/b tuning affects ranking, and how FieldBm25Index weights
//! different document fields differently.

use rag::keyword::{Bm25Config, Bm25Index, FieldBm25Index};
use rag::vector_store::Document;

fn main() {
    let docs = vec![
        Document::new("Rust programming language safety systems".to_string()),
        Document::new("Python easy scripting dynamic typing".to_string()),
        Document::new("Rust ownership borrowing memory safe".to_string()),
        Document::new("Go programming language concurrency channels".to_string()),
    ];

    // Default BM25 (k1=1.2, b=0.75)
    let default_idx = Bm25Index::from_documents(&docs).unwrap();
    let default_hits = default_idx.search("rust programming", 3);
    println!("Default BM25 hits for 'rust programming':");
    for (id, score) in &default_hits {
        let doc = docs.iter().find(|d| d.id == *id).unwrap();
        println!("  {:.3}  {}", score, doc.content);
    }

    // Tuned BM25 (k1=1.5, b=0.5) - less length normalization
    let tuned_idx = Bm25Index::from_documents_with_config(&docs, Bm25Config::new(1.5, 0.5)).unwrap();
    let tuned_hits = tuned_idx.search("rust programming", 3);
    println!("\nTuned BM25 (k1=1.5, b=0.5) hits for 'rust programming':");
    for (id, score) in &tuned_hits {
        let doc = docs.iter().find(|d| d.id == *id).unwrap();
        println!("  {:.3}  {}", score, doc.content);
    }

    // Field-level boosts: title weighted 3x, body 1x
    let mut field_docs = Vec::new();
    for i in 0..3 {
        let mut d = Document::new(format!("content{}", i));
        d.metadata.insert("title".to_string(), "rust programming guide".to_string());
        d.metadata.insert("body".to_string(), "detailed explanation of concepts".to_string());
        field_docs.push(d);
    }
    let mut d = Document::new("content3".to_string());
    d.metadata.insert("title".to_string(), "other topic".to_string());
    d.metadata.insert("body".to_string(), "rust programming deep dive and examples".to_string());
    field_docs.push(d);

    let mut fidx = FieldBm25Index::new(vec![
        ("title".to_string(), 3.0),
        ("body".to_string(), 1.0),
    ]);
    fidx.build(&field_docs).unwrap();

    println!("\nField-level BM25 (title=3x, body=1x) for 'rust programming':");
    let field_hits = fidx.search("rust programming", 4);
    for (id, score) in &field_hits {
        let doc = field_docs.iter().find(|d| d.id == *id).unwrap();
        let title = doc.metadata.get("title").unwrap();
        let body = doc.metadata.get("body").unwrap();
        println!("  {:.3}  title='{}' body='{}'", score, title, body);
    }
}
