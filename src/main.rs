use clap::{Parser, Subcommand};
use rag::{
    chunker::{FixedSizeChunker, ParagraphChunker},
    embeddings::{OllamaEmbeddingModel, OpenAIEmbeddingModel},
    graph::GraphStore,
    graph_rag::{GraphRagEngine, SimpleEntityExtractor},
    retriever::Retriever,
    vector_store::{InMemoryVectorStore, JsonPersistentVectorStore, VectorStore},
};
use std::path::PathBuf;
use tokio::fs;

#[derive(Parser)]
#[command(name = "rag")]
#[command(about = "RAG CLI: vector + BM25 hybrid and optional graph snapshot tools", long_about = None)]
struct Cli {
    /// Directory for `vectors.json`, `graph.json`, and `graph_rag.json`.
    #[arg(long, env = "RAG_STATE_DIR", default_value = ".rag", global = true)]
    state_dir: PathBuf,

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
    /// Dense vector search only.
    Query {
        #[arg(short, long)]
        query: String,

        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
    /// Vector + BM25 merge (library-aligned hybrid retrieval).
    HybridQuery {
        #[arg(short, long)]
        query: String,

        #[arg(short, long, default_value_t = 5)]
        top_k: usize,

        /// Weight on the vector channel in [0, 1]; lexical weight is `1 - alpha`.
        #[arg(short, long, default_value_t = 0.6)]
        alpha: f32,
    },
    List {
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        #[arg(short, long, default_value_t = 0)]
        offset: usize,
    },
    Count,
    /// Load `graph.json` (or `--graph`) and print stats.
    GraphStats {
        #[arg(short, long)]
        graph: Option<PathBuf>,
    },
    /// Ingest a file into GraphRAG and save `graph_rag.json` (+ `graph.json`).
    GraphBuild {
        #[arg(short, long)]
        file: PathBuf,

        #[arg(short, long, default_value = "document")]
        source: String,
    },
    /// Run hybrid graph+vector query from `graph_rag.json` snapshot.
    GraphHybridQuery {
        #[arg(short, long)]
        query: String,

        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}

async fn ensure_state_dir(dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dir).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let api_key = std::env::var("OPENAI_API_KEY").ok();
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let ollama_model =
        std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    match cli.command {
        Commands::Add { file, source } => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            let store = JsonPersistentVectorStore::open(&store_path).await?;
            let content = fs::read_to_string(&file).await?;
            println!("Adding document: {}", file.display());

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(ParagraphChunker))
                    .with_top_k(5);

                let doc_ids = retriever
                    .add_document_with_metadata(
                        content,
                        vec![
                            ("source".to_string(), source.clone()),
                            ("path".to_string(), file.display().to_string()),
                        ],
                    )
                    .await?;

                println!("Document added successfully. Chunk IDs: {}", doc_ids);
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(ParagraphChunker))
                    .with_top_k(5);

                let doc_ids = retriever
                    .add_document_with_metadata(
                        content,
                        vec![
                            ("source".to_string(), source.clone()),
                            ("path".to_string(), file.display().to_string()),
                        ],
                    )
                    .await?;

                println!("Document added successfully. Chunk IDs: {}", doc_ids);
            }
            println!("State saved under {}", store_path.display());
        }
        Commands::Query { query, top_k } => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            if !store_path.exists() {
                eprintln!("No index at {}. Run `rag add` first.", store_path.display());
                return Ok(());
            }
            let store = JsonPersistentVectorStore::open(&store_path).await?;
            println!("Query: {}", query);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                print_results(retriever.retrieve_with_scores(&query).await?);
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                print_results(retriever.retrieve_with_scores(&query).await?);
            }
        }
        Commands::HybridQuery { query, top_k, alpha } => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            if !store_path.exists() {
                eprintln!("No index at {}. Run `rag add` first.", store_path.display());
                return Ok(());
            }
            let store = JsonPersistentVectorStore::open(&store_path).await?;
            println!("Hybrid query (alpha={}): {}", alpha, query);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                print_results(retriever.retrieve_hybrid(&query, alpha).await?);
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(Box::new(FixedSizeChunker::new(500, 50)))
                    .with_top_k(top_k);

                print_results(retriever.retrieve_hybrid(&query, alpha).await?);
            }
        }
        Commands::List { limit, offset } => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            if !store_path.exists() {
                println!("No index yet.");
                return Ok(());
            }
            let store = JsonPersistentVectorStore::open(&store_path).await?;
            let documents = store.list(limit, offset).await?;
            let total = store.count().await?;

            println!("Showing {} documents (total: {}):", documents.len(), total);
            for (i, doc) in documents.iter().enumerate() {
                println!("{}. ID: {}", i + 1 + offset, doc.id);
                println!(
                    "   Content: {}...",
                    doc.content.chars().take(100).collect::<String>()
                );
                if !doc.metadata.is_empty() {
                    println!("   Metadata: {:?}", doc.metadata);
                }
                println!();
            }
        }
        Commands::Count => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            if !store_path.exists() {
                println!("Total documents in store: 0");
                return Ok(());
            }
            let store = JsonPersistentVectorStore::open(&store_path).await?;
            let count = store.count().await?;
            println!("Total documents in store: {}", count);
        }
        Commands::GraphStats { graph } => {
            let path = graph.unwrap_or_else(|| cli.state_dir.join("graph.json"));
            if !path.exists() {
                eprintln!("Graph file not found: {}", path.display());
                return Ok(());
            }
            let g = GraphStore::load_from_file(&path)?;
            let comm = g.detect_communities();
            println!(
                "Graph {} — nodes: {}, edges: {}, density: {:.4}, communities: {}",
                path.display(),
                g.node_count(),
                g.edge_count(),
                g.density(),
                comm.len()
            );
        }
        Commands::GraphBuild { file, source } => {
            ensure_state_dir(&cli.state_dir).await?;
            let content = fs::read_to_string(&file).await?;
            let rag_path = cli.state_dir.join("graph_rag.json");
            let graph_path = cli.state_dir.join("graph.json");

            if let Some(key) = api_key {
                let embed = OpenAIEmbeddingModel::new(key);
                let engine = GraphRagEngine::new(
                    SimpleEntityExtractor::new(),
                    embed,
                    InMemoryVectorStore::new(),
                )
                .with_chunker(Box::new(ParagraphChunker));
                engine.add_document(content).await?;
                engine.save_snapshot(&rag_path).await?;
                engine.graph_store().save_to_file(&graph_path)?;
            } else {
                let embed = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let engine = GraphRagEngine::new(
                    SimpleEntityExtractor::new(),
                    embed,
                    InMemoryVectorStore::new(),
                )
                .with_chunker(Box::new(ParagraphChunker));
                engine.add_document(content).await?;
                engine.save_snapshot(&rag_path).await?;
                engine.graph_store().save_to_file(&graph_path)?;
            }
            println!(
                "Wrote {} and {} (source metadata: {})",
                rag_path.display(),
                graph_path.display(),
                source
            );
        }
        Commands::GraphHybridQuery { query, top_k } => {
            let rag_path = cli.state_dir.join("graph_rag.json");
            if !rag_path.exists() {
                eprintln!(
                    "Missing {}. Run `rag graph-build` first.",
                    rag_path.display()
                );
                return Ok(());
            }

            if let Some(key) = api_key {
                let embed = OpenAIEmbeddingModel::new(key);
                let engine =
                    GraphRagEngine::load_from_snapshot_file(&rag_path, SimpleEntityExtractor::new(), embed)
                        .await?
                        .with_top_k(top_k);
                print_graph_results(engine.query(&query).await?);
            } else {
                let embed = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let engine =
                    GraphRagEngine::load_from_snapshot_file(&rag_path, SimpleEntityExtractor::new(), embed)
                        .await?
                        .with_top_k(top_k);
                print_graph_results(engine.query(&query).await?);
            }
        }
    }

    Ok(())
}

fn print_results(results: Vec<(String, f32)>) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }
    println!("\nFound {} chunks:\n", results.len());
    for (i, (content, score)) in results.iter().enumerate() {
        println!("{}. Score: {:.4}", i + 1, score);
        println!("   {}\n", content);
    }
}

fn print_graph_results(results: Vec<rag::GraphRagResult>) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }
    println!("\nFound {} results:\n", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("{}. [{}] score={:.4}", i + 1, r.source, r.score);
        println!("   {}\n", r.content);
        if !r.entities.is_empty() {
            println!("   entities: {:?}\n", r.entities);
        }
    }
}
