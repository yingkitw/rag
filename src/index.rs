use crate::vector_store::{Document, Similarity};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Supported distance metrics for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_distance). Range: [-1, 1].
    /// Most common for text embeddings. Higher is more similar.
    #[default]
    Cosine,
    /// Euclidean distance. Range: [0, +inf). Lower is more similar.
    /// We return negative distance so higher is more similar (consistent with Cosine).
    Euclidean,
    /// Dot product. Range: (-inf, +inf). Higher is more similar.
    /// Best for normalized vectors.
    DotProduct,
    /// Manhattan distance (L1). Range: [0, +inf). Lower is more similar.
    /// We return negative distance so higher is more similar.
    Manhattan,
}

impl DistanceMetric {
    /// Compute similarity between two vectors using this metric.
    /// Higher value always means more similar (consistent across all metrics).
    /// Dispatches to SIMD-accelerated kernels when available.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        match self {
            DistanceMetric::Cosine => crate::simd::cosine_similarity(a, b),
            DistanceMetric::Euclidean => {
                let dist = crate::simd::euclidean_distance(a, b);
                if dist == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + dist)
                }
            }
            DistanceMetric::DotProduct => crate::simd::dot_product(a, b),
            DistanceMetric::Manhattan => {
                let dist = crate::simd::manhattan_distance(a, b);
                if dist == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + dist)
                }
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::DotProduct => "dot_product",
            DistanceMetric::Manhattan => "manhattan",
        }
    }
}

/// Trait for vector search indexes.
/// Implementations can range from brute-force flat indexes to approximate
/// nearest neighbor structures like HNSW, IVF, etc.
pub trait Index: Send + Sync {
    /// Add a document to the index.
    fn add(&self, document: Document);

    /// Remove a document from the index by ID.
    fn remove(&self, id: &str) -> bool;

    /// Search for the top-k most similar documents to the query vector.
    fn search(&self, query: &[f32], top_k: usize) -> Vec<Similarity>;

    /// Exact kNN search over documents matching a predicate (exhaustive, not approximate).
    fn search_exact_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &dyn Fn(&Document) -> bool,
    ) -> Vec<Similarity>;

    /// Batch search: find top-k for each query vector.
    fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<Similarity>> {
        queries
            .iter()
            .map(|q| self.search(q, top_k))
            .collect()
    }

    /// Clear all documents from the index.
    fn clear(&self);

    /// Return the number of indexed documents.
    fn len(&self) -> usize;

    /// Return true if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the expected vector dimension, if known.
    fn dimension(&self) -> Option<usize>;

    /// Return the distance metric used by this index.
    fn metric(&self) -> DistanceMetric;
}

/// Brute-force flat index. Exact search, no approximation.
/// Suitable for small-to-medium datasets (< 100k documents).
/// Uses parallel search for better performance.
pub struct FlatIndex {
    documents: dashmap::DashMap<String, Arc<Document>>,
    metric: DistanceMetric,
    dimension: Option<usize>,
}

impl FlatIndex {
    pub fn new() -> Self {
        Self {
            documents: dashmap::DashMap::new(),
            metric: DistanceMetric::default(),
            dimension: None,
        }
    }

    pub fn with_metric(metric: DistanceMetric) -> Self {
        Self {
            documents: dashmap::DashMap::new(),
            metric,
            dimension: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            documents: dashmap::DashMap::with_capacity(capacity),
            metric: DistanceMetric::default(),
            dimension: None,
        }
    }

}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for FlatIndex {
    fn add(&self, document: Document) {
        self.documents
            .insert(document.id.clone(), Arc::new(document));
    }

    fn remove(&self, id: &str) -> bool {
        self.documents.remove(id).is_some()
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<Similarity> {
        flat_search_top_k(&self.documents, query, self.metric, top_k, &|_| true)
    }

    fn search_exact_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &dyn Fn(&Document) -> bool,
    ) -> Vec<Similarity> {
        flat_search_top_k(&self.documents, query, self.metric, top_k, filter)
    }

    fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<Similarity>> {
        use std::sync::Arc as StdArc;
        use std::thread;

        let num_queries = queries.len();
        if num_queries == 0 {
            return Vec::new();
        }

        // For small batches, sequential is faster (avoids thread overhead)
        if num_queries <= 4 || self.documents.len() < 1000 {
            return queries
                .iter()
                .map(|q| self.search(q, top_k))
                .collect();
        }

        // Collect all docs once and share via Arc across threads
        let docs: StdArc<Vec<StdArc<Document>>> = StdArc::new(
            self.documents.iter().map(|entry| entry.value().clone()).collect()
        );
        let metric = self.metric;

        let mut handles = Vec::with_capacity(num_queries);
        #[allow(clippy::unnecessary_to_owned)]
        for query in queries.iter().cloned() {
            let docs = StdArc::clone(&docs);
            let handle = thread::spawn(move || {
                flat_search_top_k_slice(&docs, &query, metric, top_k, &|_| true)
            });
            handles.push(handle);
        }

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_default())
            .collect()
    }

    fn clear(&self) {
        self.documents.clear();
    }

    fn len(&self) -> usize {
        self.documents.len()
    }

    fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[derive(Clone)]
pub(crate) struct HeapItem {
    score: f32,
    doc: Arc<Document>,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score)
    }
}

/// Find the top-k most similar documents using a bounded min-heap.
/// Avoids sorting all candidates and cloning documents beyond the top-k.
pub(crate) fn flat_search_top_k(
    documents: &dashmap::DashMap<String, Arc<Document>>,
    query: &[f32],
    metric: DistanceMetric,
    top_k: usize,
    filter: &dyn Fn(&Document) -> bool,
) -> Vec<Similarity> {
    if top_k == 0 || documents.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(top_k);
    for entry in documents.iter() {
        let doc = entry.value();
        if !filter(doc) {
            continue;
        }
        if let Some(emb) = &doc.embedding {
            let score = metric.similarity(query, emb);
            if heap.len() < top_k {
                heap.push(Reverse(HeapItem {
                    score,
                    doc: Arc::clone(doc),
                }));
            } else if let Some(Reverse(top)) = heap.peek() && score > top.score {
                heap.pop();
                heap.push(Reverse(HeapItem {
                    score,
                    doc: Arc::clone(doc),
                }));
            }
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|Reverse(item)| Similarity {
            document: (*item.doc).clone(),
            score: item.score,
        })
        .collect()
}

/// Heap-based top-k over a pre-collected slice of documents (used by batch search).
pub(crate) fn flat_search_top_k_slice(
    docs: &[Arc<Document>],
    query: &[f32],
    metric: DistanceMetric,
    top_k: usize,
    filter: &dyn Fn(&Document) -> bool,
) -> Vec<Similarity> {
    if top_k == 0 || docs.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(top_k);
    for doc in docs {
        if !filter(doc) {
            continue;
        }
        if let Some(emb) = &doc.embedding {
            let score = metric.similarity(query, emb);
            if heap.len() < top_k {
                heap.push(Reverse(HeapItem {
                    score,
                    doc: Arc::clone(doc),
                }));
            } else if let Some(Reverse(top)) = heap.peek() && score > top.score {
                heap.pop();
                heap.push(Reverse(HeapItem {
                    score,
                    doc: Arc::clone(doc),
                }));
            }
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|Reverse(item)| Similarity {
            document: (*item.doc).clone(),
            score: item.score,
        })
        .collect()
}

/// Vector utilities for normalization and validation.
pub mod utils {
    /// L2-normalize a vector in-place.
    pub fn l2_normalize(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Return a new L2-normalized vector.
    pub fn l2_normalize_copy(vector: &[f32]) -> Vec<f32> {
        let mut v = vector.to_vec();
        l2_normalize(&mut v);
        v
    }

    /// Validate that all vectors in a batch have the same dimension.
    pub fn validate_dimensions(vectors: &[Vec<f32>]) -> crate::errors::Result<usize> {
        if vectors.is_empty() {
            return Err(crate::errors::RagError::EmbeddingError(
                "Empty vector batch".to_string(),
            ));
        }
        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(crate::errors::RagError::EmbeddingError(format!(
                    "Vector {} has dimension {} (expected {})",
                    i,
                    v.len(),
                    dim
                )));
            }
        }
        Ok(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((DistanceMetric::Cosine.similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(DistanceMetric::Cosine.similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((DistanceMetric::Euclidean.similarity(&a, &b) - 1.0).abs() < 1e-6);
        let sim = DistanceMetric::Euclidean.similarity(&a, &c);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 4 + 10 + 18 = 32
        assert!((DistanceMetric::DotProduct.similarity(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((DistanceMetric::Manhattan.similarity(&a, &b) - 1.0).abs() < 1e-6);
        let sim = DistanceMetric::Manhattan.similarity(&a, &c);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_flat_index_search() {
        let index = FlatIndex::new();
        let doc1 = Document::new("doc1".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let doc2 = Document::new("doc2".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);
        let doc3 = Document::new("doc3".to_string()).with_embedding(vec![0.9, 0.1, 0.0]);

        index.add(doc1.clone());
        index.add(doc2.clone());
        index.add(doc3.clone());

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, doc1.id);
        assert_eq!(results[1].document.id, doc3.id);
    }

    #[test]
    fn test_flat_index_batch_search() {
        let index = FlatIndex::new();
        let doc1 = Document::new("doc1".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let doc2 = Document::new("doc2".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);

        index.add(doc1);
        index.add(doc2);

        let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let results = index.search_batch(&queries, 1);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[1].len(), 1);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        utils::l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        utils::l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_copy() {
        let v = vec![6.0, 8.0];
        let normalized = utils::l2_normalize_copy(&v);
        assert_eq!(v, vec![6.0, 8.0]); // original unchanged
        let norm = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flat_index_empty_search() {
        let index = FlatIndex::new();
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_flat_index_remove() {
        let index = FlatIndex::new();
        let doc = Document::new("test".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let id = doc.id.clone();

        index.add(doc);
        assert_eq!(index.len(), 1);

        let removed = index.remove(&id);
        assert!(removed);
        assert_eq!(index.len(), 0);

        let not_removed = index.remove("nonexistent");
        assert!(!not_removed);
    }

    #[test]
    fn test_flat_index_clear() {
        let index = FlatIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));
        index.add(Document::new("b".to_string()).with_embedding(vec![0.0, 1.0]));
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_flat_index_document_without_embedding() {
        let index = FlatIndex::new();
        let doc = Document::new("no embedding".to_string());
        index.add(doc);

        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_flat_index_top_k_larger_than_data() {
        let index = FlatIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));

        let results = index.search(&[1.0, 0.0], 100);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_flat_index_search_zero_top_k() {
        let index = FlatIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));

        let results = index.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_flat_index_batch_search_empty() {
        let index = FlatIndex::new();
        let results = index.search_batch(&[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_flat_index_with_capacity() {
        let index = FlatIndex::with_capacity(100);
        assert!(index.is_empty());
    }

    #[test]
    fn test_distance_metric_names() {
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
        assert_eq!(DistanceMetric::Euclidean.name(), "euclidean");
        assert_eq!(DistanceMetric::DotProduct.name(), "dot_product");
        assert_eq!(DistanceMetric::Manhattan.name(), "manhattan");
    }

    #[test]
    fn test_default_distance_metric_is_cosine() {
        let metric: DistanceMetric = Default::default();
        assert_eq!(metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_validate_dimensions_ok() {
        let vectors = vec![vec![1.0; 128], vec![2.0; 128], vec![3.0; 128]];
        let dim = utils::validate_dimensions(&vectors).unwrap();
        assert_eq!(dim, 128);
    }

    #[test]
    fn test_validate_dimensions_mismatch() {
        let vectors = vec![vec![1.0; 128], vec![2.0; 64]];
        let result = utils::validate_dimensions(&vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimensions_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let result = utils::validate_dimensions(&vectors);
        assert!(result.is_err());
    }
}
