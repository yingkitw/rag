//! Example: Contextual retrieval - rewrite chunks with document context.
//!
//! This demonstrates the API. To run with a real LLM, set OPENAI_API_KEY.

use rag::contextual::ContextualRetrieval;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("Note: Set OPENAI_API_KEY to use the live API.");
        println!("Running in demo mode (no actual API call).\n");
        "demo".to_string()
    });

    let rewriter = ContextualRetrieval::openai(api_key);

    let chunk = "The borrow checker ensures references are valid.";
    let document_context = "This document explains Rust's ownership system, which includes the borrow checker, lifetimes, and move semantics.";

    println!("Original chunk:\n  {}\n", chunk);
    println!("Document context:\n  {}\n", document_context);

    if std::env::var("OPENAI_API_KEY").is_ok() {
        match rewriter.rewrite(chunk, document_context).await {
            Ok(rewritten) => {
                println!("Rewritten chunk:\n  {}", rewritten);
            }
            Err(e) => {
                eprintln!("API error: {}", e);
            }
        }
    } else {
        println!("Rewritten chunk (demo):");
        println!("  In Rust's ownership system, the borrow checker ensures that references are always valid, preventing dangling pointers and data races at compile time.");
    }
}
