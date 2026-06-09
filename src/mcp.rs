use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::*,
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
    ServerHandler
};
use serde::Deserialize;
use serde_json::json;

use crate::chunker::{ParagraphChunker, TextChunker};
use crate::embeddings::{EmbeddingModel, OllamaEmbeddingModel, OpenAIEmbeddingModel};
use crate::graph::{GraphEdge, GraphNode, GraphStore};
use crate::graph_rag::{EntityExtractor, SimpleEntityExtractor};
use crate::vector_store::{InMemoryVectorStore, VectorStore};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddDocumentParams {
    #[schemars(description = "The text content of the document to add")]
    pub content: String,
    #[schemars(description = "Optional source identifier for the document")]
    pub source: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct QueryParams {
    #[schemars(description = "The search query text")]
    pub query: String,
    #[schemars(description = "Number of top results to return (default: 5)")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListDocumentsParams {
    #[schemars(description = "Maximum number of documents to return (default: 10)")]
    pub limit: Option<usize>,
    #[schemars(description = "Number of documents to skip (default: 0)")]
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetEntityParams {
    #[schemars(description = "Name of the entity to look up")]
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetNeighborsParams {
    #[schemars(description = "Name of the entity")]
    pub name: String,
    #[schemars(description = "Traversal depth (default: 1)")]
    pub depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GraphQueryParams {
    #[schemars(description = "The search query text")]
    pub query: String,
    #[schemars(description = "Number of top results to return (default: 5)")]
    pub top_k: Option<usize>,
    #[schemars(description = "Graph traversal depth for expansion (default: 2)")]
    pub depth: Option<usize>,
}

enum EmbeddingBackend {
    OpenAI(OpenAIEmbeddingModel),
    Ollama(OllamaEmbeddingModel),
}

#[async_trait]
impl EmbeddingModel for EmbeddingBackend {
    async fn embed(&self, texts: Vec<String>) -> crate::errors::Result<Vec<Vec<f32>>> {
        match self {
            Self::OpenAI(m) => m.embed(texts).await,
            Self::Ollama(m) => m.embed(texts).await,
        }
    }

    async fn embed_single(&self, text: &str) -> crate::errors::Result<Vec<f32>> {
        match self {
            Self::OpenAI(m) => m.embed_single(text).await,
            Self::Ollama(m) => m.embed_single(text).await,
        }
    }
}

fn tool_result(json_value: serde_json::Value) -> CallToolResult {
    CallToolResult::success(vec![Content::text(json_value.to_string())])
}

fn tool_error(message: &str) -> CallToolResult {
    CallToolResult::error(vec![Content::text(
        json!({ "error": message }).to_string(),
    )])
}

#[derive(Clone)]
pub struct RagMcpServer {
    store: Arc<InMemoryVectorStore>,
    graph: Arc<GraphStore>,
    embedding: Arc<EmbeddingBackend>,
    extractor: Arc<SimpleEntityExtractor>,
    entity_chunks: Arc<DashMap<String, HashSet<String>>>,
    chunk_entities: Arc<DashMap<String, HashSet<String>>>,
    top_k: usize,
}

impl RagMcpServer {
    pub fn new_openai(api_key: String) -> Self {
        Self {
            store: Arc::new(InMemoryVectorStore::new()),
            graph: Arc::new(GraphStore::new()),
            embedding: Arc::new(EmbeddingBackend::OpenAI(OpenAIEmbeddingModel::new(api_key))),
            extractor: Arc::new(SimpleEntityExtractor::new()),
            entity_chunks: Arc::new(DashMap::new()),
            chunk_entities: Arc::new(DashMap::new()),
            top_k: 5,
        }
    }

    pub fn new_ollama(model: String, base_url: Option<String>) -> Self {
        let mut ollama = OllamaEmbeddingModel::new(model);
        if let Some(url) = base_url {
            ollama = ollama.with_base_url(url);
        }
        Self {
            store: Arc::new(InMemoryVectorStore::new()),
            graph: Arc::new(GraphStore::new()),
            embedding: Arc::new(EmbeddingBackend::Ollama(ollama)),
            extractor: Arc::new(SimpleEntityExtractor::new()),
            entity_chunks: Arc::new(DashMap::new()),
            chunk_entities: Arc::new(DashMap::new()),
            top_k: 5,
        }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    fn ensure_entity_node(&self, name: &str, label: &str) -> Option<String> {
        if let Some(existing) = self.graph.get_node_by_name(name) {
            return Some(existing.id);
        }
        let node = GraphNode::new(name.to_string(), label.to_string());
        let id = node.id.clone();
        match self.graph.add_node(node) {
            Ok(()) => Some(id),
            Err(_) => None,
        }
    }

    fn link_co_occurrence(&self, src_id: &str, tgt_id: &str) {
        if src_id == tgt_id {
            return;
        }

        if let Some(existing) = self.graph.find_edge(src_id, tgt_id, "co_occurs") {
            let updated = GraphEdge::new(
                src_id.to_string(),
                tgt_id.to_string(),
                "co_occurs".to_string(),
            )
            .with_weight(existing.weight + 1.0);
            let _ = self.graph.upsert_edge(updated);
        } else {
            let edge = GraphEdge::new(
                src_id.to_string(),
                tgt_id.to_string(),
                "co_occurs".to_string(),
            );
            let _ = self.graph.add_edge(edge);
        }

        if self.graph.find_edge(tgt_id, src_id, "co_occurs").is_none() {
            let edge = GraphEdge::new(
                tgt_id.to_string(),
                src_id.to_string(),
                "co_occurs".to_string(),
            );
            let _ = self.graph.add_edge(edge);
        }
    }
}

#[tool_router]
impl RagMcpServer {
    #[tool(description = "Add a document to the RAG vector store. The document will be chunked, embedded, and indexed for later retrieval.")]
    async fn rag_add_document(
        &self,
        Parameters(params): Parameters<AddDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let chunker = ParagraphChunker;
        let chunks: Vec<String> = chunker.chunk(&params.content).map_err(|e: crate::errors::RagError| {
            McpError::internal_error("chunk_error", Some(json!({"error": e.to_string()})))
        })?;

        let embeddings = self.embedding.embed(chunks.clone()).await.map_err(|e| {
            McpError::internal_error("embedding_error", Some(json!({"error": e.to_string()})))
        })?;

        let mut doc_ids: Vec<String> = Vec::new();
        for (chunk_text, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let mut doc = crate::vector_store::Document::new(chunk_text).with_embedding(embedding);
            if let Some(ref source) = params.source {
                doc = doc.with_metadata("source".to_string(), source.clone());
            }
            let id = doc.id.clone();
            self.store.add(doc).await.map_err(|e| {
                McpError::internal_error("store_error", Some(json!({"error": e.to_string()})))
            })?;
            doc_ids.push(id);
        }

        Ok(tool_result(json!({
            "success": true,
            "chunk_count": doc_ids.len(),
            "chunk_ids": doc_ids,
        })))
    }

    #[tool(description = "Query the RAG vector store for semantically similar documents. Returns the top-k most relevant chunks with similarity scores.")]
    async fn rag_query(
        &self,
        Parameters(params): Parameters<QueryParams>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = params.top_k.unwrap_or(self.top_k);
        let query_embedding = self
            .embedding
            .embed_single(&params.query)
            .await
            .map_err(|e| {
                McpError::internal_error(
                    "embedding_error",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

        let results = self
            .store
            .search(&query_embedding, top_k)
            .await
            .map_err(|e| {
                McpError::internal_error(
                    "search_error",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

        let results_json: Vec<serde_json::Value> = results
            .into_iter()
            .enumerate()
            .map(|(i, sim)| {
                json!({
                    "rank": i + 1,
                    "content": sim.document.content,
                    "score": format!("{:.4}", sim.score),
                    "id": sim.document.id,
                })
            })
            .collect();

        Ok(tool_result(json!({
            "query": params.query,
            "results": results_json,
        })))
    }

    #[tool(description = "List documents in the RAG vector store with pagination.")]
    async fn rag_list_documents(
        &self,
        Parameters(params): Parameters<ListDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(10);
        let offset = params.offset.unwrap_or(0);

        let documents = self
            .store
            .list(limit, offset)
            .await
            .map_err(|e| {
                McpError::internal_error("list_error", Some(json!({"error": e.to_string()})))
            })?;
        let total = self.store.count().await.map_err(|e| {
            McpError::internal_error("count_error", Some(json!({"error": e.to_string()})))
        })?;

        let docs_json: Vec<serde_json::Value> = documents
            .into_iter()
            .map(|doc| {
                json!({
                    "id": doc.id,
                    "content": doc.content.chars().take(200).collect::<String>(),
                    "metadata": doc.metadata,
                })
            })
            .collect();

        Ok(tool_result(json!({
            "documents": docs_json,
            "total": total,
        })))
    }

    #[tool(description = "Count the total number of document chunks in the RAG vector store.")]
    async fn rag_count(&self) -> Result<CallToolResult, McpError> {
        let count = self.store.count().await.map_err(|e| {
            McpError::internal_error("count_error", Some(json!({"error": e.to_string()})))
        })?;

        Ok(tool_result(json!({ "total_chunks": count })))
    }

    #[tool(description = "Build a knowledge graph from a document. Extracts entities (proper nouns, acronyms, quoted terms) and creates co-occurrence relationships between entities found in the same chunk.")]
    async fn graph_build(
        &self,
        Parameters(params): Parameters<AddDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let chunker = ParagraphChunker;
        let chunks: Vec<String> = chunker.chunk(&params.content).map_err(|e: crate::errors::RagError| {
            McpError::internal_error("chunk_error", Some(json!({"error": e.to_string()})))
        })?;

        let embeddings = self.embedding.embed(chunks.clone()).await.map_err(|e| {
            McpError::internal_error("embedding_error", Some(json!({"error": e.to_string()})))
        })?;

        let mut all_entities: Vec<String> = Vec::new();
        let mut doc_ids: Vec<String> = Vec::new();

        for (chunk_text, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let entities = self.extractor.extract_entities(&chunk_text).await;
            let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();

            let entity_ids: Vec<String> = entities
                .iter()
                .filter_map(|e| self.ensure_entity_node(&e.name, &e.label))
                .collect();

            for i in 0..entity_ids.len() {
                for j in (i + 1)..entity_ids.len() {
                    self.link_co_occurrence(&entity_ids[i], &entity_ids[j]);
                }
            }

            let mut doc =
                crate::vector_store::Document::new(chunk_text).with_embedding(embedding);
            if let Some(ref source) = params.source {
                doc = doc.with_metadata("source".to_string(), source.clone());
            }
            doc = doc.with_metadata("entities".to_string(), entity_names.join(","));
            let id = doc.id.clone();

            for name in &entity_names {
                self.entity_chunks
                    .entry(name.clone())
                    .or_insert_with(HashSet::new)
                    .insert(id.clone());

                self.chunk_entities
                    .entry(id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(name.clone());
            }

            self.store.add(doc).await.map_err(|e| {
                McpError::internal_error("store_error", Some(json!({"error": e.to_string()})))
            })?;
            doc_ids.push(id);

            for name in &entity_names {
                if !all_entities.iter().any(|e| e == name) {
                    all_entities.push(name.clone());
                }
            }
        }

        Ok(tool_result(json!({
            "success": true,
            "chunk_count": doc_ids.len(),
            "entity_count": all_entities.len(),
            "entities": all_entities,
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
        })))
    }

    #[tool(description = "Hybrid query combining vector similarity search with knowledge graph traversal. Finds relevant chunks via embeddings, then expands results through entity relationships in the graph.")]
    async fn graph_query(
        &self,
        Parameters(params): Parameters<GraphQueryParams>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = params.top_k.unwrap_or(self.top_k);
        let depth = params.depth.unwrap_or(2);

        let query_embedding = self
            .embedding
            .embed_single(&params.query)
            .await
            .map_err(|e| {
                McpError::internal_error(
                    "embedding_error",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

        let vector_results = self
            .store
            .search(&query_embedding, top_k)
            .await
            .map_err(|e| {
                McpError::internal_error(
                    "search_error",
                    Some(json!({"error": e.to_string()})),
                )
            })?;

        let query_entities = self.extractor.extract_entities(&params.query).await;
        let mut graph_chunk_ids = HashSet::new();

        for entity in &query_entities {
            if let Some(node) = self.graph.get_node_by_name(&entity.name) {
                let reachable = self.graph.bfs(&node.id, depth);
                for neighbor in &reachable {
                    if let Some(chunks) = self.entity_chunks.get(&neighbor.name) {
                        for chunk_id in chunks.value().iter() {
                            graph_chunk_ids.insert(chunk_id.clone());
                        }
                    }
                }
                if let Some(chunks) = self.entity_chunks.get(&node.name) {
                    for chunk_id in chunks.value().iter() {
                        graph_chunk_ids.insert(chunk_id.clone());
                    }
                }
            }
        }

        let mut seen_ids = HashSet::new();
        let mut results: Vec<serde_json::Value> = Vec::new();

        for sim in &vector_results {
            seen_ids.insert(sim.document.id.clone());
            let entities = self
                .chunk_entities
                .get(&sim.document.id)
                .map(|e| e.value().iter().cloned().collect::<Vec<String>>())
                .unwrap_or_default();

            results.push(json!({
                "rank": results.len() + 1,
                "content": sim.document.content,
                "score": format!("{:.4}", sim.score),
                "source": "vector",
                "entities": entities,
            }));
        }

        for chunk_id in &graph_chunk_ids {
            if seen_ids.insert((*chunk_id).clone()) {
                if let Ok(Some(doc)) = self.store.get(chunk_id).await {
                    let entities = self
                        .chunk_entities
                        .get(chunk_id)
                        .map(|e| e.value().iter().cloned().collect::<Vec<String>>())
                        .unwrap_or_default();

                    results.push(json!({
                        "rank": results.len() + 1,
                        "content": doc.content,
                        "score": "0.0000",
                        "source": "graph",
                        "entities": entities,
                    }));
                }
            }
        }

        results.truncate(top_k);

        let entity_names: Vec<&str> = query_entities.iter().map(|e| e.name.as_str()).collect();

        Ok(tool_result(json!({
            "query": params.query,
            "query_entities": entity_names,
            "results": results,
        })))
    }

    #[tool(description = "Get detailed information about an entity in the knowledge graph, including its type, connections, and associated document count.")]
    async fn graph_get_entity(
        &self,
        Parameters(params): Parameters<GetEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.graph.get_node_by_name(&params.name) {
            Some(node) => {
                let neighbors = self.graph.neighbors(&node.id);
                let neighbor_names: Vec<String> =
                    neighbors.iter().map(|n| n.name.clone()).collect();
                let degree = self.graph.degree(&node.id);
                let chunk_count = self
                    .entity_chunks
                    .get(&node.name)
                    .map(|e| e.value().len())
                    .unwrap_or(0);

                Ok(tool_result(json!({
                    "name": node.name,
                    "label": node.label,
                    "degree": degree,
                    "chunk_count": chunk_count,
                    "neighbors": neighbor_names,
                })))
            }
            None => Ok(tool_error(&format!(
                "Entity '{}' not found in the knowledge graph",
                params.name
            ))),
        }
    }

    #[tool(description = "Get the neighbors of an entity in the knowledge graph up to a specified traversal depth.")]
    async fn graph_get_neighbors(
        &self,
        Parameters(params): Parameters<GetNeighborsParams>,
    ) -> Result<CallToolResult, McpError> {
        let depth = params.depth.unwrap_or(1);

        match self.graph.get_node_by_name(&params.name) {
            Some(node) => {
                let reachable = self.graph.bfs(&node.id, depth);
                let neighbor_names: Vec<String> =
                    reachable.iter().map(|n| n.name.clone()).collect();

                Ok(tool_result(json!({
                    "entity": node.name,
                    "depth": depth,
                    "neighbor_count": neighbor_names.len(),
                    "neighbors": neighbor_names,
                })))
            }
            None => Ok(tool_error(&format!("Entity '{}' not found", params.name))),
        }
    }

    #[tool(description = "Get statistics about the knowledge graph: node count, edge count, density, and community count.")]
    async fn graph_info(&self) -> Result<CallToolResult, McpError> {
        let communities = self.graph.detect_communities();
        let doc_count = self.store.count().await.unwrap_or(0);

        Ok(tool_result(json!({
            "node_count": self.graph.node_count(),
            "edge_count": self.graph.edge_count(),
            "density": format!("{:.4}", self.graph.density()),
            "community_count": communities.len(),
            "document_chunks": doc_count,
        })))
    }

    #[tool(description = "Detect communities in the knowledge graph using label propagation. Returns groups of closely related entities.")]
    async fn graph_communities(&self) -> Result<CallToolResult, McpError> {
        let communities = self.graph.detect_communities();

        let communities_json: Vec<serde_json::Value> = communities
            .into_iter()
            .map(|c| {
                let names: Vec<String> = c
                    .node_ids
                    .iter()
                    .filter_map(|id| self.graph.get_node(id).map(|n| n.name))
                    .collect();
                json!({
                    "community_id": c.id,
                    "size": c.size,
                    "entities": names,
                })
            })
            .collect();

        Ok(tool_result(json!({ "communities": communities_json })))
    }
}

#[tool_handler(
    name = "rag-mcp-server",
    version = "0.1.0",
    instructions = "RAG MCP Server provides Retrieval-Augmented Generation and GraphRAG capabilities. Use rag_add_document to index documents, rag_query for semantic search, graph_build to construct knowledge graphs from documents, graph_query for hybrid vector+graph retrieval, and graph_get_entity/graph_get_neighbors/graph_communities to explore the knowledge graph."
)]
impl ServerHandler for RagMcpServer {}
