use clap::{Parser, Subcommand};
use rag::{
    chunker::{FixedSizeChunker, ParagraphChunker},
    embeddings::{OllamaEmbeddingModel, OpenAIEmbeddingModel},
    retriever::Retriever,
    vector_store::InMemoryVectorStore,
    vector_store::VectorStore,
    Document,
};
use std::path::PathBuf;
use tokio::fs;

#[derive(Parser)]
#[command(name = "rag")]
#[command(about = "A RAG (Retrieval-Augmented Generation) CLI tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        #[arg(short, long)]
        file: PathBuf,

        #[arg(short, long, default_value = "document")]
        source: String,
    },
    Query {
        #[arg(short, long)]
        query: String,

        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
    List {
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        #[arg(short, long, default_value_t = 0)]
        offset: usize,
    },
    Count,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let api_key = std::env::var("OPENAI_API_KEY").ok();
    let ollama_url = std::env::var("OLLAMA_URL").unwrap_or("http://localhost:11434".to_string());

    let model_name = if api_key.is_some() { "OpenAI" } else { "Ollama" };
    println!("Using embedding model: {}", model_name);

    let vector_store = InMemoryVectorStore::new();

    match cli.command {
        Commands::Add { file, source } => {
            let content = fs::read_to_string(&file).await?;
            println!("Adding document: {}", file.display());

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, InMemoryVectorStore::new())
                    .with_chunker(Box::new(ParagraphChunker))
                    .with_top_k(5);

                let doc_ids = retriever
                    .add_document_with_metadata(
                        content,
                        vec![("source".to_string(), source.clone()), ("path".to_string(), file.display().to_string())],
                    )
                    .await?;

                println!("Document added successfully. Chunk IDs: {}", doc_ids);
                println!("Source: {}", source);
            } else {
                let embedding_model = OllamaEmbeddingModel::new("nomic-embed-text".to_string()).with_base_url(ollama_url);
                let retriever = Retriever::new(embedding_model, InMemoryVectorStore::new())
                    .with_chunker(Box::new(ParagraphChunker))
                    .with_top_k(5);

                let doc_ids = retriever
                    .add_document_with_metadata(
                        content,
                        vec![("source".to_string(), source.clone()), ("path".to_string(), file.display().to_string())],
                    )
                    .await?;

                println!("Document added successfully. Chunk IDs: {}", doc_ids);
                println!("Source: {}", source);
            }
        }
        Commands::Query { query, top_k } => {
            println!("Query: {}", query);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, InMemoryVectorStore::new())
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                let results = retriever.retrieve_with_scores(&query).await?;

                if results.is_empty() {
                    println!("No results found.");
                } else {
                    println!("\nFound {} relevant chunks:\n", results.len());
                    for (i, (content, score)) in results.iter().enumerate() {
                        println!("{}. Score: {:.4}", i + 1, score);
                        println!("   {}\n", content);
                    }
                }
            } else {
                let embedding_model = OllamaEmbeddingModel::new("nomic-embed-text".to_string()).with_base_url(ollama_url);
                let retriever = Retriever::new(embedding_model, InMemoryVectorStore::new())
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                let results = retriever.retrieve_with_scores(&query).await?;

                if results.is_empty() {
                    println!("No results found.");
                } else {
                    println!("\nFound {} relevant chunks:\n", results.len());
                    for (i, (content, score)) in results.iter().enumerate() {
                        println!("{}. Score: {:.4}", i + 1, score);
                        println!("   {}\n", content);
                    }
                }
            }
        }
        Commands::List { limit, offset } => {
            let documents = vector_store.list(limit, offset).await?;
            let total = vector_store.count().await?;

            println!("Showing {} documents (total: {}):", documents.len(), total);
            for (i, doc) in documents.iter().enumerate() {
                println!("{}. ID: {}", i + 1 + offset, doc.id);
                println!("   Content: {}...", doc.content.chars().take(100).collect::<String>());
                if !doc.metadata.is_empty() {
                    println!("   Metadata: {:?}", doc.metadata);
                }
                println!();
            }
        }
        Commands::Count => {
            let count = vector_store.count().await?;
            println!("Total documents in store: {}", count);
        }
    }

    Ok(())
}