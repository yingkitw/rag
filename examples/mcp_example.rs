use rag::{
    mcp::{McpRequest, McpServer},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RAG MCP Server Example\n");

    let mcp_server = McpServer::new_ollama();

    println!("1. Initializing MCP server...");
    let init_req = McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: "initialize".to_string(),
        params: None,
    };

    let init_response = mcp_server.handle_request(init_req).await;
    println!("Initialize response: {}\n", serde_json::to_string_pretty(&init_response)?);

    println!("2. Listing available tools...");
    let list_req = McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(2)),
        method: "tools/list".to_string(),
        params: None,
    };

    let list_response = mcp_server.handle_request(list_req).await;
    println!("Tools list response: {}\n", serde_json::to_string_pretty(&list_response)?);

    println!("3. Adding a document...");
    let add_req = McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(3)),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": "rag_add_document",
            "arguments": {
                "content": "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by providing them with relevant external context.",
                "source": "example-doc"
            }
        })),
    };

    let add_response = mcp_server.handle_request(add_req).await;
    println!("Add document response: {}\n", serde_json::to_string_pretty(&add_response)?);

    println!("4. Querying the document...");
    let query_req = McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(4)),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": "rag_query",
            "arguments": {
                "query": "What is Rust?",
                "top_k": 3
            }
        })),
    };

    let query_response = mcp_server.handle_request(query_req).await;
    println!("Query response: {}\n", serde_json::to_string_pretty(&query_response)?);

    println!("5. Counting documents...");
    let count_req = McpRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(5)),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": "rag_count",
            "arguments": {}
        })),
    };

    let count_response = mcp_server.handle_request(count_req).await;
    println!("Count response: {}\n", serde_json::to_string_pretty(&count_response)?);

    Ok(())
}