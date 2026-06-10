//! Example: Combine vector search and BM25 using Reciprocal Rank Fusion (RRF).
//!
//! RRF fuses multiple ranked result lists by rank position rather than raw scores,
//! making it robust to score scale differences across search channels.

use rag::{
    hybrid::rrf_fusion,
    keyword::Bm25Index,
    vector_store::{Document, Similarity},
};

fn main() {
    let docs = vec![
        Document::new("Rust is a systems programming language with memory safety".to_string()),
        Document::new("Python is easy to learn and great for scripting".to_string()),
        Document::new("Rust uses ownership and borrowing for safe concurrency".to_string()),
        Document::new("JavaScript runs in browsers and on servers with Node.js".to_string()),
    ];

    // Simulate vector search results (high scores for semantic similarity)
    let vector_results = vec![
        Similarity { document: docs[2].clone(), score: 0.92 }, // rust concurrency
        Similarity { document: docs[0].clone(), score: 0.85 }, // rust systems
        Similarity { document: docs[1].clone(), score: 0.40 }, // python
        Similarity { document: docs[3].clone(), score: 0.30 }, // js
    ];

    // BM25 keyword search results (exact word matches)
    let bm25 = Bm25Index::from_documents(&docs).unwrap();
    let keyword_results: Vec<Similarity> = bm25
        .search("rust safety", 4)
        .into_iter()
        .map(|(id, score)| {
            let doc = docs.iter().find(|d| d.id == id).unwrap().clone();
            Similarity { document: doc, score }
        })
        .collect();

    println!("Vector results:");
    for r in &vector_results {
        println!("  {:.2} - {}", r.score, r.document.content);
    }

    println!("\nBM25 results:");
    for r in &keyword_results {
        println!("  {:.2} - {}", r.score, r.document.content);
    }

    // Fuse with RRF (rank_constant = 60)
    let fused = rrf_fusion(&[vector_results, keyword_results], 60, 4);

    println!("\nRRF fused results:");
    for (i, r) in fused.iter().enumerate() {
        println!("  #{}  score={:.4}  {}", i + 1, r.score, r.document.content);
    }
}
