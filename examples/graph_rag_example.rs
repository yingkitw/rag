//! Hybrid Graph RAG: chunk, embed, vector index, and entity co-occurrence graph.
//! Requires a running Ollama with an embedding model (default: `nomic-embed-text`).
//!
//! Run: `cargo run --example graph_rag_example`
//!
//! Optional: `OLLAMA_URL`, same as other examples.

use rag::{
    chunker::FixedSizeChunker,
    embeddings::OllamaEmbeddingModel,
    graph_rag::{GraphRagEngine, SimpleEntityExtractor},
    vector_store::InMemoryVectorStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GraphRagEngine example (Ollama embeddings)\n");

    let ollama_url = std::env::var("OLLAMA_URL").ok();
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    let embedder = if let Some(url) = ollama_url {
        OllamaEmbeddingModel::new(model.clone()).with_base_url(url)
    } else {
        OllamaEmbeddingModel::new(model.clone())
    };

    let engine = GraphRagEngine::new(
        SimpleEntityExtractor::new(),
        embedder,
        InMemoryVectorStore::new(),
    )
    .with_chunker(Box::new(FixedSizeChunker::new(280, 40)))
    .with_top_k(5)
    .with_graph_depth(2);

    let docs = [
        "Alice owns the RAG design at Acme Corp. NLP embeddings power retrieval.",
        "Bob pairs with Alice on the same Acme RAG release train.",
        "Unrelated note: cosine similarity compares normalized vectors for semantic search.",
    ];

    println!("Ingesting {} documents...", docs.len());
    for (i, text) in docs.iter().enumerate() {
        match engine.add_document(text.to_string()).await {
            Ok(ids) => println!("  doc {} -> {} chunks", i + 1, ids.len()),
            Err(e) => {
                eprintln!("Ollama/embed failed: {e}");
                eprintln!("Skipping remainder (start Ollama or set OLLAMA_URL).");
                return Ok(());
            }
        }
    }

    let info = engine.graph_info();
    println!(
        "\nGraph: {} nodes, {} edges, {} communities (density {:.4})",
        info.node_count,
        info.edge_count,
        info.community_count,
        info.density
    );

    if let Some(e) = engine.get_entity_info("Alice") {
        println!(
            "\nEntity Alice: label={}, degree={}, chunk_count={}, neighbors={:?}",
            e.label, e.degree, e.chunk_count, e.neighbors
        );
    }

    let queries = ["What does Bob do with Alice?", "How is RAG used at Acme?"];
    for q in queries {
        println!("\nQuery: {q}");
        let results = engine.query(q).await?;
        for (i, r) in results.iter().enumerate() {
            println!(
                "  {}. [{} score={:.4}] {}",
                i + 1,
                r.source,
                r.score,
                r.content.chars().take(120).collect::<String>()
            );
            if !r.entities.is_empty() {
                println!("      entities: {:?}", r.entities);
            }
        }
    }

    Ok(())
}
