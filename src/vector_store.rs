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
    #[serde(default)]
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

#[allow(async_fn_in_trait)]
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
        let metric = self.index.metric();
        let mut similarities: Vec<Similarity> = self
            .documents
            .iter()
            .filter(|entry| filter.matches(&entry.value().metadata))
            .filter_map(|entry| {
                let doc = entry.value();
                doc.embedding.as_ref().map(|emb| Similarity {
                    document: doc.clone(),
                    score: metric.similarity(query, emb),
                })
            })
            .collect();
        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        Ok(similarities)
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
            .skip(offset)
            .take(limit)
            .cloned()
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

/// Load every stored document (paged via `list`).
pub async fn load_all_documents<S: VectorStore>(store: &S) -> Result<Vec<Document>> {
    let n = store.count().await?;
    if n == 0 {
        return Ok(Vec::new());
    }
    store.list(n, 0).await
}

/// [`InMemoryVectorStore`] that flushes JSON to disk after each mutating operation.
/// For better performance, use [`open_lazy_flush`](Self::open_lazy_flush) and call [`flush`](Self::flush) manually.
pub struct JsonPersistentVectorStore {
    path: std::path::PathBuf,
    inner: InMemoryVectorStore,
    auto_flush: bool,
}

impl JsonPersistentVectorStore {
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            InMemoryVectorStore::load_from_file(&path).await?
        } else {
            InMemoryVectorStore::new()
        };
        Ok(Self { path, inner, auto_flush: true })
    }

    pub async fn open_with_metric<P: AsRef<Path>>(path: P, metric: DistanceMetric) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            InMemoryVectorStore::load_from_file(&path).await?
        } else {
            InMemoryVectorStore::with_metric(metric)
        };
        Ok(Self { path, inner, auto_flush: true })
    }

    /// Open without auto-flushing. Call [`flush`](Self::flush) manually or on drop.
    pub async fn open_lazy_flush<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            InMemoryVectorStore::load_from_file(&path).await?
        } else {
            InMemoryVectorStore::new()
        };
        Ok(Self { path, inner, auto_flush: false })
    }

    pub async fn flush(&self) -> Result<()> {
        self.inner.save_to_file(&self.path).await
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl VectorStore for JsonPersistentVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        self.inner.add(document).await?;
        if self.auto_flush { self.flush().await } else { Ok(()) }
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        self.inner.add_batch(documents).await?;
        if self.auto_flush { self.flush().await } else { Ok(()) }
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Similarity>> {
        self.inner.search(query, top_k).await
    }

    async fn search_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<Similarity>> {
        self.inner.search_with_filter(query, top_k, filter).await
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        self.inner.search_batch(queries, top_k).await
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        self.inner.get(id).await
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let ok = self.inner.delete(id).await?;
        if ok && self.auto_flush {
            self.flush().await?;
        }
        Ok(ok)
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let n = self.inner.delete_batch(ids).await?;
        if n > 0 && self.auto_flush {
            self.flush().await?;
        }
        Ok(n)
    }

    async fn clear(&self) -> Result<()> {
        self.inner.clear().await?;
        if self.auto_flush { self.flush().await } else { Ok(()) }
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        self.inner.list(limit, offset).await
    }

    async fn count(&self) -> Result<usize> {
        self.inner.count().await
    }

    fn metric(&self) -> DistanceMetric {
        self.inner.metric()
    }
}