use rag::mcp::RagMcpServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RAG MCP Server Example\n");
    println!("This example demonstrates creating an rmcp-based MCP server.");
    println!("For a real MCP server, use: rag-mcp binary\n");

    let _server = RagMcpServer::new_ollama("nomic-embed-text".to_string(), None);

    println!("Server created successfully.");
    println!("Available tools:");
    println!("  - rag_add_document: Add documents to vector store");
    println!("  - rag_query: Semantic search");
    println!("  - rag_list_documents: List stored documents");
    println!("  - rag_count: Count documents");
    println!("  - graph_build: Build knowledge graph from text");
    println!("  - graph_query: Hybrid vector+graph retrieval");
    println!("  - graph_get_entity: Get entity details");
    println!("  - graph_get_neighbors: Explore entity relationships");
    println!("  - graph_info: Graph statistics");
    println!("  - graph_communities: Detect entity communities");

    println!("\nTo use this server with an MCP client, run:");
    println!("  cargo run --bin rag-mcp");

    Ok(())
}
