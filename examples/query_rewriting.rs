//! Example: Query rewriting - generate search query variants with an LLM.
//!
//! This demonstrates the API. To run with a real LLM, set OPENAI_API_KEY.

use rag::query_rewriting::QueryRewriter;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("Note: Set OPENAI_API_KEY to use the live API.");
        println!("Running in demo mode (no actual API call).\n");
        "demo".to_string()
    });

    let rewriter = QueryRewriter::openai(api_key);
    let query = "how do I handle errors in rust";

    println!("Original query: {}\n", query);

    if std::env::var("OPENAI_API_KEY").is_ok() {
        match rewriter.rewrite(query, 3).await {
            Ok(variants) => {
                println!("Rewritten variants:");
                for (i, v) in variants.iter().enumerate() {
                    println!("  {}. {}", i + 1, v);
                }
            }
            Err(e) => {
                eprintln!("API error: {}", e);
            }
        }
    } else {
        println!("Rewritten variants (demo):");
        println!("  1. Rust error handling best practices");
        println!("  2. How to use Result and Option in Rust");
        println!("  3. Rust panic and error propagation patterns");
    }
}
