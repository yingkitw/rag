use crate::errors::Result;
use crate::index::{DistanceMetric, FlatIndex, Index};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl Document {
    pub fn new(content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    pub fn with_id(id: String, content: String) -> Self {
        Self {
            id,
            content,
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Similarity {
    pub document: Document,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    pub filters: Vec<(String, String)>,
}

impl MetadataFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, key: String, value: String) -> Self {
        self.filters.push((key, value));
        self
    }

    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        if self.filters.is_empty() {
            return true;
        }

        for (key, value) in &self.filters {
            if !metadata.get(key).map(|v| v == value).unwrap_or(false) {
                return false;
            }
        }

        true
    }
}

pub trait VectorStore: Send + Sync {
    async fn add(&self, document: Document) -> Result<()>;
    async fn add_batch(&self, documents: Vec<Document>) -> Result<()>;
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Similarity>>;
    async fn search_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<Similarity>>;
    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>>;
    async fn get(&self, id: &str) -> Result<Option<Document>>;
    async fn delete(&self, id: &str) -> Result<bool>;
    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize>;
    async fn clear(&self) -> Result<()>;
    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>>;
    async fn count(&self) -> Result<usize>;
    fn metric(&self) -> DistanceMetric;
}

/// Compute cosine similarity between two vectors.
/// Deprecated: use [`DistanceMetric::Cosine`] instead.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    DistanceMetric::Cosine.similarity(a, b)
}

pub struct InMemoryVectorStore {
    index: FlatIndex,
    documents: dashmap::DashMap<String, Document>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            index: FlatIndex::new(),
            documents: dashmap::DashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: FlatIndex::with_capacity(capacity),
            documents: dashmap::DashMap::with_capacity(capacity),
        }
    }

    pub fn with_metric(metric: DistanceMetric) -> Self {
        Self {
            index: FlatIndex::with_metric(metric),
            documents: dashmap::DashMap::new(),
        }
    }

    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let docs_vec: Vec<Document> = self.documents.iter().map(|entry| entry.value().clone()).collect();

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &docs_vec)?;

        Ok(())
    }

    pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let docs_vec: Vec<Document> = serde_json::from_str(&content)?;

        let store = Self::new();
        for doc in docs_vec {
            store.index.add(doc.clone());
            store.documents.insert(doc.id.clone(), doc);
        }

        Ok(store)
    }
}

impl VectorStore for InMemoryVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        let id = document.id.clone();
        self.index.add(document.clone());
        self.documents.insert(id, document);
        Ok(())
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        for doc in documents {
            let id = doc.id.clone();
            self.index.add(doc.clone());
            self.documents.insert(id, doc);
        }
        Ok(())
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
        let results = self.index.search(query, top_k * 4);
        let filtered: Vec<Similarity> = results
            .into_iter()
            .filter(|s| filter.matches(&s.document.metadata))
            .take(top_k)
            .collect();
        Ok(filtered)
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        Ok(self.index.search_batch(queries, top_k))
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        Ok(self.documents.get(id).map(|entry| entry.value().clone()))
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let removed = self.documents.remove(id).is_some();
        if removed {
            self.index.remove(id);
        }
        Ok(removed)
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let mut count = 0;
        for id in ids {
            if self.documents.remove(&id).is_some() {
                self.index.remove(&id);
                count += 1;
            }
        }
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        self.documents.clear();
        self.index.clear();
        Ok(())
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        Ok(self
            .documents
            .iter()
            .skip(offset)
            .take(limit)
            .map(|entry| entry.value().clone())
            .collect())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.documents.len())
    }

    fn metric(&self) -> DistanceMetric {
        self.index.metric()
    }
}

pub struct MinimalVectorDB {
    index: FlatIndex,
    documents: Arc<RwLock<HashMap<String, Document>>>,
}

impl Default for MinimalVectorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl MinimalVectorDB {
    pub fn new() -> Self {
        Self {
            index: FlatIndex::new(),
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: FlatIndex::with_capacity(capacity),
            documents: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
        }
    }

    pub fn with_metric(metric: DistanceMetric) -> Self {
        Self {
            index: FlatIndex::with_metric(metric),
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let docs = self.documents.read().unwrap();
        let docs_vec: Vec<Document> = docs.values().cloned().collect();

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &docs_vec)?;

        Ok(())
    }

    pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let docs_vec: Vec<Document> = serde_json::from_str(&content)?;

        let mut docs = HashMap::new();
        let index = FlatIndex::new();
        for doc in docs_vec {
            index.add(doc.clone());
            docs.insert(doc.id.clone(), doc);
        }

        Ok(Self {
            index,
            documents: Arc::new(RwLock::new(docs)),
        })
    }
}

impl VectorStore for MinimalVectorDB {
    async fn add(&self, document: Document) -> Result<()> {
        let id = document.id.clone();
        self.index.add(document.clone());
        let mut docs = self.documents.write().unwrap();
        docs.insert(id, document);
        Ok(())
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        for doc in documents {
            let id = doc.id.clone();
            self.index.add(doc.clone());
            docs.insert(id, doc);
        }
        Ok(())
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
        let results = self.index.search(query, top_k * 4);
        let filtered: Vec<Similarity> = results
            .into_iter()
            .filter(|s| filter.matches(&s.document.metadata))
            .take(top_k)
            .collect();
        Ok(filtered)
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        Ok(self.index.search_batch(queries, top_k))
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        let docs = self.documents.read().unwrap();
        Ok(docs.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let removed = {
            let mut docs = self.documents.write().unwrap();
            docs.remove(id).is_some()
        };
        if removed {
            self.index.remove(id);
        }
        Ok(removed)
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let mut count = 0;
        for id in ids {
            let removed = {
                let mut docs = self.documents.write().unwrap();
                docs.remove(&id).is_some()
            };
            if removed {
                self.index.remove(&id);
                count += 1;
            }
        }
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        docs.clear();
        self.index.clear();
        Ok(())
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        let docs = self.documents.read().unwrap();
        Ok(docs
            .values()
            .cloned()
            .skip(offset)
            .take(limit)
            .collect())
    }

    async fn count(&self) -> Result<usize> {
        let docs = self.documents.read().unwrap();
        Ok(docs.len())
    }

    fn metric(&self) -> DistanceMetric {
        self.index.metric()
    }
}