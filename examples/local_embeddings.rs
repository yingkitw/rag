//! Example: Local ONNX embeddings + cross-encoder reranker (requires `fastembed`).
//! Models download to a local cache on first run, then run fully offline.

#[cfg(feature = "fastembed")]
use rag::fastembed_store::{FastEmbedEmbeddingModel, FastEmbedReranker};

#[cfg(feature = "fastembed")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rag::EmbeddingModel;
    use rag::rerank::SimilarityReranker;
    use rag::vector_store::{Document, Similarity};

    let model = FastEmbedEmbeddingModel::new();

    let texts = vec![
        "Rust provides memory safety without a garbage collector".to_string(),
        "The Eiffel Tower is in Paris".to_string(),
        "Ownership and borrowing are core to Rust".to_string(),
    ];

    let embeddings = model.embed(texts.clone()).await?;
    println!(
        "Embedded {} texts -> dim {}",
        embeddings.len(),
        embeddings[0].len()
    );

    // Rerank candidate chunks with a local cross-encoder.
    let reranker = FastEmbedReranker::new();
    let items = texts
        .into_iter()
        .map(|t| Similarity {
            document: Document::new(t),
            score: 0.0,
        })
        .collect::<Vec<_>>();
    let reranked = reranker
        .rerank("How does Rust handle memory?", items)
        .await?;
    println!("\nReranked results:");
    for r in &reranked {
        println!("  {:.4} - {}", r.score, r.document.content);
    }
    Ok(())
}

#[cfg(not(feature = "fastembed"))]
fn main() {
    eprintln!("Rebuild with: cargo run --example local_embeddings --features fastembed");
}
