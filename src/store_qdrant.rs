//! Qdrant-backed vector store implementing [`VectorStore`].
//!
//! Talks to a Qdrant server via its REST API. The collection must already
//! exist and be configured with the correct vector size and distance metric.

use crate::errors::{RagError, Result};
use crate::index::DistanceMetric;
use crate::vector_store::{Document, MetadataFilter, Similarity, VectorStore};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct QdrantVectorStore {
    client: Client,
    base_url: String,
    collection: String,
    api_key: Option<String>,
    metric: DistanceMetric,
}

impl QdrantVectorStore {
    pub fn new(base_url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            collection: collection.into(),
            api_key: None,
            metric: DistanceMetric::Cosine,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    fn url(&self, path: &str) -> String {
        format!("{}/collections/{}{}", self.base_url, self.collection, path)
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(key) = &self.api_key {
            headers.insert(
                "api-key",
                key.parse().unwrap(),
            );
        }
        headers
    }

    async fn upsert_points(&self, points: Vec<QdrantPoint>) -> Result<()> {
        let body = json!({ "points": points });
        let resp = self
            .client
            .put(self.url("/points"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant upsert failed: {}", e)))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant upsert error: {}", text)));
        }
        Ok(())
    }

    async fn scroll_ids(&self, limit: usize, offset: Option<String>) -> Result<(Vec<String>, Option<String>)> {
        let mut body = json!({ "limit": limit, "with_payload": false, "with_vector": false });
        if let Some(off) = offset {
            body["offset"] = json!(off);
        }
        let resp = self
            .client
            .post(self.url("/points/scroll"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant scroll failed: {}", e)))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant scroll error: {}", text)));
        }
        let scroll: QdrantScroll = resp
            .json()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant scroll decode failed: {}", e)))?;
        let ids: Vec<String> = scroll.result.into_iter().map(|p| p.id.to_string()).collect();
        let next_offset = scroll.next_page_offset.map(|v| v.to_string());
        Ok((ids, next_offset))
    }

    fn doc_to_point(doc: &Document) -> QdrantPoint {
        QdrantPoint {
            id: QdrantId::String(doc.id.clone()),
            vector: doc.embedding.clone().unwrap_or_default(),
            payload: {
                let mut map = serde_json::Map::new();
                map.insert("content".to_string(), json!(doc.content));
                map.insert("metadata".to_string(), json!(doc.metadata));
                map
            },
        }
    }

    fn retrieved_to_doc(point: &QdrantRetrievedPoint) -> Option<Document> {
        let content = point.payload.get("content")?.as_str()?.to_string();
        let metadata: std::collections::HashMap<String, String> = point
            .payload
            .get("metadata")?
            .as_object()
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let embedding = point.vector.clone().filter(|v| !v.is_empty());
        Some(Document {
            id: point.id.to_string(),
            content,
            metadata,
            embedding,
        })
    }

    fn scored_to_doc(point: QdrantScoredPoint) -> Option<Document> {
        let content = point.payload.get("content")?.as_str()?.to_string();
        let metadata: std::collections::HashMap<String, String> = point
            .payload
            .get("metadata")?
            .as_object()
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let embedding = if point.vector.is_empty() { None } else { Some(point.vector) };
        Some(Document {
            id: point.id.to_string(),
            content,
            metadata,
            embedding,
        })
    }

    fn build_filter(filter: &MetadataFilter) -> Option<Value> {
        if filter.filters.is_empty() {
            return None;
        }
        let conditions: Vec<Value> = filter
            .filters
            .iter()
            .map(|(k, v)| {
                json!({
                    "key": k,
                    "match": { "value": v }
                })
            })
            .collect();
        Some(json!({ "must": conditions }))
    }
}

#[allow(async_fn_in_trait)]
impl VectorStore for QdrantVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        self.upsert_points(vec![Self::doc_to_point(&document)]).await
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        let points: Vec<_> = documents.iter().map(Self::doc_to_point).collect();
        self.upsert_points(points).await
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Similarity>> {
        self.search_with_filter(query, top_k, &MetadataFilter::new()).await
    }

    async fn search_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<Similarity>> {
        let mut body = json!({
            "vector": query,
            "limit": top_k,
            "with_payload": true,
            "with_vector": true,
        });
        if let Some(f) = Self::build_filter(filter) {
            body["filter"] = f;
        }
        let resp = self
            .client
            .post(self.url("/points/search"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant search failed: {}", e)))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant search error: {}", text)));
        }
        let result: QdrantSearchResult = resp
            .json()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant search decode failed: {}", e)))?;
        let similarities = result
            .result
            .into_iter()
            .filter_map(|p| {
                let score = p.point.score;
                Self::scored_to_doc(p.point).map(|doc| Similarity {
                    document: doc,
                    score,
                })
            })
            .collect();
        Ok(similarities)
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.search(q, top_k).await?);
        }
        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        let resp = self
            .client
            .get(self.url(&format!("/points/{}", id)))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant get failed: {}", e)))?;
        if resp.status().as_u16() == 404 {
            return Ok(None);
        }
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant get error: {}", text)));
        }
        let result: QdrantPointResult = resp
            .json()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant get decode failed: {}", e)))?;
        Ok(Self::retrieved_to_doc(&result.point))
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let body = json!({ "points": [id] });
        let resp = self
            .client
            .post(self.url("/points/delete"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant delete failed: {}", e)))?;
        Ok(resp.status().is_success())
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let body = json!({ "points": ids });
        let resp = self
            .client
            .post(self.url("/points/delete"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant delete batch failed: {}", e)))?;
        Ok(if resp.status().is_success() { ids.len() } else { 0 })
    }

    async fn clear(&self) -> Result<()> {
        let mut all_ids = Vec::new();
        let mut offset = None;
        loop {
            let (ids, next) = self.scroll_ids(1000, offset).await?;
            if ids.is_empty() {
                break;
            }
            all_ids.extend(ids);
            if next.is_none() {
                break;
            }
            offset = next;
        }
        if !all_ids.is_empty() {
            self.delete_batch(all_ids).await?;
        }
        Ok(())
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        let body = json!({
            "limit": limit,
            "offset": offset,
            "with_payload": true,
            "with_vector": true,
        });
        let resp = self
            .client
            .post(self.url("/points/scroll"))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant list failed: {}", e)))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant list error: {}", text)));
        }
        let scroll: QdrantScroll = resp
            .json()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant list decode failed: {}", e)))?;
        let docs: Vec<Document> = scroll
            .result
            .into_iter()
            .filter_map(|p| Self::retrieved_to_doc(&p))
            .collect();
        Ok(docs)
    }

    async fn count(&self) -> Result<usize> {
        let resp = self
            .client
            .post(self.url("/points/count"))
            .headers(self.headers())
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant count failed: {}", e)))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RagError::VectorStoreError(format!("Qdrant count error: {}", text)));
        }
        let count: QdrantCount = resp
            .json()
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Qdrant count decode failed: {}", e)))?;
        Ok(count.result.count as usize)
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

// ------------------------------------------------------------------
// Qdrant JSON types
// ------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct QdrantPoint {
    id: QdrantId,
    vector: Vec<f32>,
    payload: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct QdrantPointResult {
    #[serde(flatten)]
    point: QdrantRetrievedPoint,
}

#[derive(Debug, Deserialize)]
struct QdrantRetrievedPoint {
    id: QdrantId,
    payload: serde_json::Map<String, Value>,
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct QdrantScoredPoint {
    id: QdrantId,
    score: f32,
    payload: serde_json::Map<String, Value>,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResult {
    result: Vec<QdrantSearchHit>,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchHit {
    #[serde(flatten)]
    point: QdrantScoredPoint,
}

#[derive(Debug, Deserialize)]
struct QdrantScroll {
    result: Vec<QdrantRetrievedPoint>,
    next_page_offset: Option<QdrantId>,
}

#[derive(Debug, Deserialize)]
struct QdrantCount {
    result: QdrantCountResult,
}

#[derive(Debug, Deserialize)]
struct QdrantCountResult {
    count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum QdrantId {
    String(String),
    Integer(i64),
    Uuid(String),
}

impl std::fmt::Display for QdrantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QdrantId::String(s) => write!(f, "{}", s),
            QdrantId::Integer(i) => write!(f, "{}", i),
            QdrantId::Uuid(u) => write!(f, "{}", u),
        }
    }
}
