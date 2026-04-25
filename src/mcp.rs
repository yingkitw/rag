use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chunker::ParagraphChunker;
use crate::embeddings::{EmbeddingModel, OllamaEmbeddingModel, OpenAIEmbeddingModel};
use crate::retriever::Retriever;
use crate::vector_store::{InMemoryVectorStore, VectorStore};

#[derive(Debug, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub result: Option<serde_json::Value>,
    pub error: Option<McpError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

pub enum McpServer {
    OpenAI(Retriever<OpenAIEmbeddingModel, InMemoryVectorStore>),
    Ollama(Retriever<OllamaEmbeddingModel, InMemoryVectorStore>),
}

impl McpServer {
    pub fn new_openai(api_key: String) -> Self {
        let embedding_model = OpenAIEmbeddingModel::new(api_key);
        let vector_store = InMemoryVectorStore::new();
        let retriever = Retriever::new(embedding_model, vector_store)
            .with_chunker(Box::new(ParagraphChunker))
            .with_top_k(5);
        Self::OpenAI(retriever)
    }

    pub fn new_ollama() -> Self {
        let embedding_model = OllamaEmbeddingModel::new("nomic-embed-text".to_string());
        let vector_store = InMemoryVectorStore::new();
        let retriever = Retriever::new(embedding_model, vector_store)
            .with_chunker(Box::new(ParagraphChunker))
            .with_top_k(5);
        Self::Ollama(retriever)
    }

    pub async fn handle_request(&self, request: McpRequest) -> McpResponse {
        let result: std::result::Result<serde_json::Value, McpError> = match request.method.as_str() {
            "initialize" => self.initialize(request.params).await,
            "tools/list" => self.list_tools().await,
            "tools/call" => self.call_tool(request.params).await,
            "ping" => Ok(json!({"status": "ok"})),
            _ => Err(McpError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
            }),
        };

        match result {
            Ok(data) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(data),
                error: None,
            },
            Err(error) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(error),
            },
        }
    }

    async fn initialize(&self, _params: Option<serde_json::Value>) -> std::result::Result<serde_json::Value, McpError> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "rag-mcp-server",
                "version": "0.1.0"
            },
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            }
        }))
    }

    async fn list_tools(&self) -> std::result::Result<serde_json::Value, McpError> {
        Ok(json!({
            "tools": [
                {
                    "name": "rag_add_document",
                    "description": "Add a document to the RAG vector store",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The document content to add"
                            },
                            "source": {
                                "type": "string",
                                "description": "Optional source identifier for the document"
                            }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "rag_query",
                    "description": "Query the RAG vector store for relevant documents",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "top_k": {
                                "type": "number",
                                "description": "Number of results to return (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "rag_list_documents",
                    "description": "List documents in the vector store",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of documents to return"
                            },
                            "offset": {
                                "type": "number",
                                "description": "Number of documents to skip"
                            }
                        }
                    }
                },
                {
                    "name": "rag_count",
                    "description": "Count total documents in the vector store",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }))
    }

    async fn call_tool(&self, params: Option<serde_json::Value>) -> std::result::Result<serde_json::Value, McpError> {
        let params = params.ok_or_else(|| McpError {
            code: -32602,
            message: "Missing params".to_string(),
        })?;

        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError {
                code: -32602,
                message: "Missing tool name".to_string(),
            })?;

        let arguments = params.get("arguments");

        match tool_name {
            "rag_add_document" => self.tool_add_document(arguments).await,
            "rag_query" => self.tool_query(arguments).await,
            "rag_list_documents" => self.tool_list_documents(arguments).await,
            "rag_count" => self.tool_count().await,
            _ => Err(McpError {
                code: -32601,
                message: format!("Unknown tool: {}", tool_name),
            }),
        }
    }

    async fn tool_add_document(&self, args: Option<&serde_json::Value>) -> std::result::Result<serde_json::Value, McpError> {
        let args = args.ok_or_else(|| McpError {
            code: -32602,
            message: "Missing arguments".to_string(),
        })?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError {
                code: -32602,
                message: "Missing content".to_string(),
            })?;

        let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("unknown");

        let metadata = vec![("source".to_string(), source.to_string())];

        let doc_ids = match self {
            Self::OpenAI(retriever) => {
                retriever
                    .add_document_with_metadata(content.to_string(), metadata)
                    .await
            }
            Self::Ollama(retriever) => {
                retriever
                    .add_document_with_metadata(content.to_string(), metadata)
                    .await
            }
        }
        .map_err(|e| McpError {
            code: -32603,
            message: format!("Failed to add document: {}", e),
        })?;

        Ok(json!({
            "success": true,
            "message": "Document added successfully",
            "chunk_ids": doc_ids
        }))
    }

    async fn tool_query(&self, args: Option<&serde_json::Value>) -> std::result::Result<serde_json::Value, McpError> {
        let args = args.ok_or_else(|| McpError {
            code: -32602,
            message: "Missing arguments".to_string(),
        })?;

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError {
                code: -32602,
                message: "Missing query".to_string(),
            })?;

        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let embedding = match self {
            Self::OpenAI(retriever) => {
                retriever.embedding_model().embed_single(query).await
            }
            Self::Ollama(retriever) => {
                retriever.embedding_model().embed_single(query).await
            }
        }
        .map_err(|e| McpError {
            code: -32603,
            message: format!("Embedding generation failed: {}", e),
        })?;

        let results_vec: Vec<crate::vector_store::Similarity> = match self {
            Self::OpenAI(retriever) => {
                retriever.vector_store().search(&embedding, top_k).await
            }
            Self::Ollama(retriever) => {
                retriever.vector_store().search(&embedding, top_k).await
            }
        }
        .map_err(|e| McpError {
            code: -32603,
            message: format!("Search failed: {}", e),
        })?;

        let results_json: Vec<_> = results_vec
            .into_iter()
            .enumerate()
            .map(|(i, similarity)| {
                json!({
                    "rank": i + 1,
                    "content": similarity.document.content,
                    "score": similarity.score
                })
            })
            .collect();

        Ok(json!({
            "query": query,
            "results": results_json
        }))
    }

    async fn tool_list_documents(&self, args: Option<&serde_json::Value>) -> std::result::Result<serde_json::Value, McpError> {
        let limit = args
            .and_then(|a| a.get("limit"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let offset = args
            .and_then(|a| a.get("offset"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let documents = match self {
            Self::OpenAI(retriever) => {
                retriever.vector_store().list(limit, offset).await
            }
            Self::Ollama(retriever) => {
                retriever.vector_store().list(limit, offset).await
            }
        }
        .map_err(|e| McpError {
            code: -32603,
            message: format!("Failed to list documents: {}", e),
        })?;

        let docs_json: Vec<_> = documents
            .into_iter()
            .map(|doc| {
                json!({
                    "id": doc.id,
                    "content": doc.content.chars().take(200).collect::<String>() + "...",
                    "metadata": doc.metadata
                })
            })
            .collect();

        Ok(json!({
            "documents": docs_json
        }))
    }

    async fn tool_count(&self) -> std::result::Result<serde_json::Value, McpError> {
        let count = match self {
            Self::OpenAI(retriever) => {
                retriever.vector_store().count().await
            }
            Self::Ollama(retriever) => {
                retriever.vector_store().count().await
            }
        }
        .map_err(|e| McpError {
            code: -32603,
            message: format!("Failed to count documents: {}", e),
        })?;

        Ok(json!({
            "total_documents": count
        }))
    }
}

impl From<crate::errors::RagError> for McpError {
    fn from(err: crate::errors::RagError) -> Self {
        McpError {
            code: -32603,
            message: err.to_string(),
        }
    }
}