use clap::{Parser, Subcommand};
use rag::{
    chunker::{
        FixedSizeChunker, ParagraphChunker, RecursiveChunker, SemanticChunker, SentenceChunker,
        StructuralChunker, TextChunker,
    },
    embeddings::{OllamaEmbeddingModel, OpenAIEmbeddingModel},
    graph::GraphStore,
    graph_rag::{GraphRagEngine, SimpleEntityExtractor},
    index::DistanceMetric,
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

    /// Chunking strategy for ingestion and queries.
    #[arg(long, global = true, default_value = "paragraph")]
    chunker: String,

    /// Distance metric for vector search.
    #[arg(long, global = true, default_value = "cosine")]
    metric: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        /// File(s) to ingest. Repeatable. Accepts files or directories.
        #[arg(short, long, num_args = 1..)]
        file: Vec<PathBuf>,

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

fn make_chunker(name: &str) -> Box<dyn TextChunker> {
    match name {
        "fixed" => Box::new(FixedSizeChunker::new(500, 50)),
        "paragraph" => Box::new(ParagraphChunker),
        "sentence" => Box::new(SentenceChunker::default()),
        "recursive" => Box::new(RecursiveChunker::default()),
        "semantic" => Box::new(SemanticChunker::default()),
        "structural" => Box::new(StructuralChunker::default()),
        _ => Box::new(ParagraphChunker),
    }
}

fn parse_metric(name: &str) -> DistanceMetric {
    match name {
        "euclidean" => DistanceMetric::Euclidean,
        "dot" | "dot_product" => DistanceMetric::DotProduct,
        "manhattan" => DistanceMetric::Manhattan,
        _ => DistanceMetric::Cosine,
    }
}

fn collect_input_paths(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for path in files {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file()
                        && let Some(ext) = p.extension()
                    {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if ext == "txt" || ext == "md" {
                            result.push(p);
                        }
                    }
                }
            }
        } else if path.is_file() {
            result.push(path.clone());
        }
    }
    result
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
            let metric = parse_metric(&cli.metric);
            let store = JsonPersistentVectorStore::open_with_metric(&store_path, metric).await?;
            let paths = collect_input_paths(&file);
            if paths.is_empty() {
                eprintln!("No files to ingest.");
                return Ok(());
            }
            let chunker = make_chunker(&cli.chunker);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
                    .with_top_k(5);
                for path in &paths {
                    let content = fs::read_to_string(path).await?;
                    println!("Adding document: {}", path.display());
                    let doc_ids = retriever
                        .add_document_with_metadata(
                            content,
                            vec![
                                ("source".to_string(), source.clone()),
                                ("path".to_string(), path.display().to_string()),
                            ],
                        )
                        .await?;
                    println!("  Chunk IDs: {}", doc_ids);
                }
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
                    .with_top_k(5);
                for path in &paths {
                    let content = fs::read_to_string(path).await?;
                    println!("Adding document: {}", path.display());
                    let doc_ids = retriever
                        .add_document_with_metadata(
                            content,
                            vec![
                                ("source".to_string(), source.clone()),
                                ("path".to_string(), path.display().to_string()),
                            ],
                        )
                        .await?;
                    println!("  Chunk IDs: {}", doc_ids);
                }
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
            let metric = parse_metric(&cli.metric);
            let store = JsonPersistentVectorStore::open_with_metric(&store_path, metric).await?;
            println!("Query: {}", query);
            let chunker = make_chunker(&cli.chunker);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
                    .with_top_k(top_k);

                print_results(retriever.retrieve_with_scores(&query).await?);
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
                    .with_top_k(top_k);

                print_results(retriever.retrieve_with_scores(&query).await?);
            }
        }
        Commands::HybridQuery {
            query,
            top_k,
            alpha,
        } => {
            ensure_state_dir(&cli.state_dir).await?;
            let store_path = cli.state_dir.join("vectors.json");
            if !store_path.exists() {
                eprintln!("No index at {}. Run `rag add` first.", store_path.display());
                return Ok(());
            }
            let metric = parse_metric(&cli.metric);
            let store = JsonPersistentVectorStore::open_with_metric(&store_path, metric).await?;
            println!("Hybrid query (alpha={}): {}", alpha, query);
            let chunker = make_chunker(&cli.chunker);

            if let Some(key) = api_key {
                let embedding_model = OpenAIEmbeddingModel::new(key);
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
                    .with_top_k(top_k);

                print_results(retriever.retrieve_hybrid(&query, alpha).await?);
            } else {
                let embedding_model = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let retriever = Retriever::new(embedding_model, store)
                    .with_chunker(chunker)
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
            let metric = parse_metric(&cli.metric);
            let store = JsonPersistentVectorStore::open_with_metric(&store_path, metric).await?;
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
            let metric = parse_metric(&cli.metric);
            let store = JsonPersistentVectorStore::open_with_metric(&store_path, metric).await?;
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
            let metadata = vec![
                ("source".to_string(), source.clone()),
                ("path".to_string(), file.display().to_string()),
            ];

            if let Some(key) = api_key {
                let embed = OpenAIEmbeddingModel::new(key);
                let engine = if rag_path.exists() {
                    println!("Loading existing snapshot from {}", rag_path.display());
                    GraphRagEngine::load_from_snapshot_file(
                        &rag_path,
                        SimpleEntityExtractor::new(),
                        embed,
                    )
                    .await?
                    .with_chunker(make_chunker(&cli.chunker))
                } else {
                    GraphRagEngine::new(
                        SimpleEntityExtractor::new(),
                        embed,
                        InMemoryVectorStore::new(),
                    )
                    .with_chunker(make_chunker(&cli.chunker))
                };
                engine.add_document_with_metadata(content, metadata).await?;
                engine.save_snapshot(&rag_path).await?;
                engine.graph_store().save_to_file(&graph_path)?;
            } else {
                let embed = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let engine = if rag_path.exists() {
                    println!("Loading existing snapshot from {}", rag_path.display());
                    GraphRagEngine::load_from_snapshot_file(
                        &rag_path,
                        SimpleEntityExtractor::new(),
                        embed,
                    )
                    .await?
                    .with_chunker(make_chunker(&cli.chunker))
                } else {
                    GraphRagEngine::new(
                        SimpleEntityExtractor::new(),
                        embed,
                        InMemoryVectorStore::new(),
                    )
                    .with_chunker(make_chunker(&cli.chunker))
                };
                engine.add_document_with_metadata(content, metadata).await?;
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
                let engine = GraphRagEngine::load_from_snapshot_file(
                    &rag_path,
                    SimpleEntityExtractor::new(),
                    embed,
                )
                .await?
                .with_top_k(top_k);
                print_graph_results(engine.query(&query).await?);
            } else {
                let embed = OllamaEmbeddingModel::new(ollama_model.clone())
                    .with_base_url(ollama_url.clone());
                let engine = GraphRagEngine::load_from_snapshot_file(
                    &rag_path,
                    SimpleEntityExtractor::new(),
                    embed,
                )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_chunker() {
        let _ = make_chunker("fixed");
        let _ = make_chunker("paragraph");
        let _ = make_chunker("sentence");
        let _ = make_chunker("recursive");
        let _ = make_chunker("semantic");
        let _ = make_chunker("structural");
        let _ = make_chunker("unknown");
    }

    #[test]
    fn test_parse_metric() {
        assert_eq!(parse_metric("cosine"), DistanceMetric::Cosine);
        assert_eq!(parse_metric("euclidean"), DistanceMetric::Euclidean);
        assert_eq!(parse_metric("dot"), DistanceMetric::DotProduct);
        assert_eq!(parse_metric("dot_product"), DistanceMetric::DotProduct);
        assert_eq!(parse_metric("manhattan"), DistanceMetric::Manhattan);
        assert_eq!(parse_metric("unknown"), DistanceMetric::Cosine);
    }

    #[test]
    fn test_collect_input_paths_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let paths = collect_input_paths(&[tmp.path().to_path_buf()]);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_collect_input_paths_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("b.md"), "world").unwrap();
        std::fs::write(dir.path().join("c.rs"), "code").unwrap();
        let paths = collect_input_paths(&[dir.path().to_path_buf()]);
        assert_eq!(paths.len(), 2);
        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"a.txt".to_string()));
        assert!(names.contains(&"b.md".to_string()));
        assert!(!names.contains(&"c.rs".to_string()));
    }

    #[test]
    fn test_cli_add_single_file() {
        let cli =
            Cli::try_parse_from(["rag", "add", "--file", "doc.txt", "--source", "test"]).unwrap();
        match cli.command {
            Commands::Add { file, source } => {
                assert_eq!(file.len(), 1);
                assert_eq!(file[0], PathBuf::from("doc.txt"));
                assert_eq!(source, "test");
            }
            _ => panic!("expected Add command"),
        }
    }

    #[test]
    fn test_cli_add_multiple_files() {
        let cli = Cli::try_parse_from([
            "rag", "add", "--file", "a.txt", "--file", "b.txt", "--source", "batch",
        ])
        .unwrap();
        match cli.command {
            Commands::Add { file, source } => {
                assert_eq!(file.len(), 2);
                assert_eq!(source, "batch");
            }
            _ => panic!("expected Add command"),
        }
    }

    #[test]
    fn test_cli_query() {
        let cli =
            Cli::try_parse_from(["rag", "query", "--query", "hello", "--top-k", "3"]).unwrap();
        match cli.command {
            Commands::Query { query, top_k } => {
                assert_eq!(query, "hello");
                assert_eq!(top_k, 3);
            }
            _ => panic!("expected Query command"),
        }
    }

    #[test]
    fn test_cli_global_flags() {
        let cli = Cli::try_parse_from([
            "rag",
            "--chunker",
            "sentence",
            "--metric",
            "euclidean",
            "query",
            "--query",
            "test",
        ])
        .unwrap();
        assert_eq!(cli.chunker, "sentence");
        assert_eq!(cli.metric, "euclidean");
    }
}
