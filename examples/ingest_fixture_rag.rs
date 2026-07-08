//! Load small Markdown fixtures from `examples/fixtures/rag_sample`, then run vector RAG.
//! Uses Ollama embeddings when available (same setup as `simple_rag`).
//!
//! Run from the repo root: `cargo run --example ingest_fixture_rag`

use std::path::PathBuf;

use rag::{
    chunker::ParagraphChunker,
    embeddings::OllamaEmbeddingModel,
    ingestion::{CodebaseSource, Source},
    retriever::Retriever,
    vector_store::{InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ingest Markdown fixtures -> Retriever\n");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fixtures/rag_sample");
    let src = CodebaseSource::new(&root).with_extensions(vec!["md".to_string()]);

    let extracted = src.extract().await?;
    println!(
        "Extracted {} file(s) from {}\n",
        extracted.len(),
        root.display()
    );

    for doc in &extracted {
        println!("- {} ({} bytes)", doc.source, doc.content.len());
    }

    let ollama_url = std::env::var("OLLAMA_URL").ok();
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    let embedding_model = if let Some(url) = ollama_url {
        OllamaEmbeddingModel::new(model.clone()).with_base_url(url)
    } else {
        OllamaEmbeddingModel::new(model.clone())
    };

    let store = InMemoryVectorStore::new();
    let retriever = Retriever::new(embedding_model, store)
        .with_chunker(Box::new(ParagraphChunker))
        .with_top_k(4);

    println!("\nIndexing into vector store...");
    for doc in &extracted {
        let meta = vec![
            ("source".to_string(), doc.source.clone()),
            (
                "path".to_string(),
                doc.metadata.get("path").cloned().unwrap_or_default(),
            ),
        ];
        if retriever
            .add_document_with_metadata(doc.content.clone(), meta)
            .await
            .is_err()
        {
            eprintln!("Embedding/index failed (is Ollama running?)");
            eprintln!("Fixtures were extracted OK; start Ollama and retry.");
            return Ok(());
        }
    }

    println!("Total chunks: {}", retriever.vector_store().count().await?);

    let query = "Who works with Alice on RAG?";
    println!("\nQuery: {query}");
    let results = retriever.retrieve_with_scores(query).await?;
    for (i, (text, score)) in results.iter().enumerate() {
        println!("  {}. [score {:.4}] {}", i + 1, score, text);
    }

    Ok(())
}
