use crate::errors::Result;
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
    async fn get(&self, id: &str) -> Result<Option<Document>>;
    async fn delete(&self, id: &str) -> Result<bool>;
    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize>;
    async fn clear(&self) -> Result<()>;
    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>>;
    async fn count(&self) -> Result<usize>;
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

pub struct InMemoryVectorStore {
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
            documents: dashmap::DashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            documents: dashmap::DashMap::with_capacity(capacity),
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
            store.documents.insert(doc.id.clone(), doc);
        }

        Ok(store)
    }
}

impl VectorStore for InMemoryVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        self.documents.insert(document.id.clone(), document);
        Ok(())
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        for doc in documents {
            self.documents.insert(doc.id.clone(), doc);
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
        let mut similarities: Vec<Similarity> = self
            .documents
            .iter()
            .filter_map(|entry| {
                let doc = entry.value();
                if let Some(embedding) = &doc.embedding {
                    if !filter.matches(&doc.metadata) {
                        return None;
                    }
                    let score = cosine_similarity(query, embedding);
                    Some(Similarity {
                        document: doc.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        similarities.truncate(top_k);

        Ok(similarities)
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        Ok(self.documents.get(id).map(|entry| entry.value().clone()))
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.documents.remove(id).is_some())
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let mut count = 0;
        for id in ids {
            if self.documents.remove(&id).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        self.documents.clear();
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
}

pub struct MinimalVectorDB {
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
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
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
        for doc in docs_vec {
            docs.insert(doc.id.clone(), doc);
        }

        Ok(Self {
            documents: Arc::new(RwLock::new(docs)),
        })
    }
}

impl VectorStore for MinimalVectorDB {
    async fn add(&self, document: Document) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        docs.insert(document.id.clone(), document);
        Ok(())
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        for doc in documents {
            docs.insert(doc.id.clone(), doc);
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
        let docs = self.documents.read().unwrap();
        let mut similarities: Vec<Similarity> = docs
            .values()
            .filter_map(|doc| {
                if let Some(embedding) = &doc.embedding {
                    if !filter.matches(&doc.metadata) {
                        return None;
                    }
                    let score = cosine_similarity(query, embedding);
                    Some(Similarity {
                        document: doc.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        similarities.truncate(top_k);

        Ok(similarities)
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        let docs = self.documents.read().unwrap();
        Ok(docs.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let mut docs = self.documents.write().unwrap();
        Ok(docs.remove(id).is_some())
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let mut docs = self.documents.write().unwrap();
        let mut count = 0;
        for id in ids {
            if docs.remove(&id).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        docs.clear();
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
}