use rmcp::{ServiceExt, transport::stdio};
use rag::mcp::RagMcpServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting RAG MCP Server");

    let api_key = std::env::var("OPENAI_API_KEY").ok();
    let ollama_url = std::env::var("OLLAMA_URL").ok();
    let ollama_model =
        std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    let server = match api_key {
        Some(key) => {
            tracing::info!("Using OpenAI embedding model");
            RagMcpServer::new_openai(key)
        }
        None => {
            tracing::info!("Using Ollama embedding model: {}", ollama_model);
            RagMcpServer::new_ollama(ollama_model, ollama_url)
        }
    };

    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Server error: {:?}", e);
    })?;

    tracing::info!("RAG MCP Server started, waiting for requests...");
    service.waiting().await?;
    Ok(())
}
