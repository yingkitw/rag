//! External reranker implementations via HTTP APIs.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::errors::{RagError, Result};
use crate::rerank::SimilarityReranker;
use crate::vector_store::Similarity;

/// Cohere reranker (requires COHERE_API_KEY env var).
pub struct CohereReranker {
    client: Client,
    api_key: String,
    model: String,
}

impl CohereReranker {
    pub fn new(api_key: String) -> Self {
        Self { client: Client::new(), api_key, model: "rerank-english-v3.0".to_string() }
    }
}

#[derive(Serialize)]
struct CohereRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct CohereResponse {
    results: Vec<CohereResult>,
}

#[derive(Deserialize)]
struct CohereResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl SimilarityReranker for CohereReranker {
    async fn rerank(&self, query: &str, items: Vec<Similarity>) -> Result<Vec<Similarity>> {
        if items.is_empty() {
            return Ok(items);
        }
        let docs: Vec<String> = items.iter().map(|i| i.document.content.clone()).collect();
        let req = CohereRequest { query: query.to_string(), documents: docs, model: self.model.clone() };
        let resp = self.client
            .post("https://api.cohere.com/v2/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send().await?;
        if !resp.status().is_success() {
            return Err(RagError::EmbeddingError(resp.text().await?));
        }
        let data: CohereResponse = resp.json().await?;
        let mut out = Vec::with_capacity(data.results.len());
        for r in data.results {
            if let Some(mut item) = items.get(r.index).cloned() {
                item.score = r.relevance_score;
                out.push(item);
            }
        }
        Ok(out)
    }
}

/// Voyage AI reranker (requires VOYAGE_API_KEY env var).
pub struct VoyageReranker {
    client: Client,
    api_key: String,
    model: String,
}

impl VoyageReranker {
    pub fn new(api_key: String) -> Self {
        Self { client: Client::new(), api_key, model: "rerank-lite-1".to_string() }
    }
}

#[derive(Serialize)]
struct VoyageRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageResult>,
}

#[derive(Deserialize)]
struct VoyageResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl SimilarityReranker for VoyageReranker {
    async fn rerank(&self, query: &str, items: Vec<Similarity>) -> Result<Vec<Similarity>> {
        if items.is_empty() {
            return Ok(items);
        }
        let docs: Vec<String> = items.iter().map(|i| i.document.content.clone()).collect();
        let req = VoyageRequest { query: query.to_string(), documents: docs, model: self.model.clone() };
        let resp = self.client
            .post("https://api.voyageai.com/v1/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send().await?;
        if !resp.status().is_success() {
            return Err(RagError::EmbeddingError(resp.text().await?));
        }
        let data: VoyageResponse = resp.json().await?;
        let mut out = Vec::with_capacity(data.data.len());
        for r in data.data {
            if let Some(mut item) = items.get(r.index).cloned() {
                item.score = r.relevance_score;
                out.push(item);
            }
        }
        Ok(out)
    }
}

/// MixedBread AI reranker (requires MIXEDBREAD_API_KEY env var).
pub struct MixedBreadReranker {
    client: Client,
    api_key: String,
    model: String,
}

impl MixedBreadReranker {
    pub fn new(api_key: String) -> Self {
        Self { client: Client::new(), api_key, model: "mixedbread-ai/mxbai-rerank-large-v1".to_string() }
    }
}

#[derive(Serialize)]
struct MixedBreadRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct MixedBreadResponse {
    data: Vec<MixedBreadResult>,
}

#[derive(Deserialize)]
struct MixedBreadResult {
    index: usize,
    score: f32,
}

#[async_trait]
impl SimilarityReranker for MixedBreadReranker {
    async fn rerank(&self, query: &str, items: Vec<Similarity>) -> Result<Vec<Similarity>> {
        if items.is_empty() {
            return Ok(items);
        }
        let docs: Vec<String> = items.iter().map(|i| i.document.content.clone()).collect();
        let req = MixedBreadRequest { query: query.to_string(), documents: docs, model: self.model.clone() };
        let resp = self.client
            .post("https://api.mixedbread.ai/v1/reranking")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send().await?;
        if !resp.status().is_success() {
            return Err(RagError::EmbeddingError(resp.text().await?));
        }
        let data: MixedBreadResponse = resp.json().await?;
        let mut out = Vec::with_capacity(data.data.len());
        for r in data.data {
            if let Some(mut item) = items.get(r.index).cloned() {
                item.score = r.score;
                out.push(item);
            }
        }
        Ok(out)
    }
}
