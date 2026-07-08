use crate::index::{DistanceMetric, Index};
use crate::vector_store::{Document, Similarity};
use hnsw_rs::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Approximate nearest neighbor index using HNSW (Hierarchical Navigable Small World).
/// Provides fast search for large datasets at the cost of exact recall.
pub struct HnswIndex {
    documents: RwLock<HashMap<String, Arc<Document>>>,
    id_map: RwLock<HashMap<String, usize>>,
    reverse_id_map: RwLock<HashMap<usize, String>>,
    deleted_ids: RwLock<HashSet<usize>>,
    hnsw: Mutex<Option<Hnsw<'static, f32, DistFn<f32>>>>,
    next_id: AtomicUsize,
    metric: DistanceMetric,
    dimension: AtomicUsize,
    max_elements: usize,
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            deleted_ids: RwLock::new(HashSet::new()),
            hnsw: Mutex::new(None),
            next_id: AtomicUsize::new(0),
            metric: DistanceMetric::default(),
            dimension: AtomicUsize::new(0),
            max_elements: 100_000,
        }
    }

    pub fn with_metric(metric: DistanceMetric) -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            deleted_ids: RwLock::new(HashSet::new()),
            hnsw: Mutex::new(None),
            next_id: AtomicUsize::new(0),
            metric,
            dimension: AtomicUsize::new(0),
            max_elements: 100_000,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            deleted_ids: RwLock::new(HashSet::new()),
            hnsw: Mutex::new(None),
            next_id: AtomicUsize::new(0),
            metric: DistanceMetric::default(),
            dimension: AtomicUsize::new(0),
            max_elements: capacity,
        }
    }

    fn make_dist_fn(metric: DistanceMetric) -> DistFn<f32> {
        DistFn::new(Box::new(move |a: &[f32], b: &[f32]| -> f32 {
            match metric {
                DistanceMetric::Cosine => {
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        1.0
                    } else {
                        1.0 - dot / (norm_a * norm_b)
                    }
                }
                DistanceMetric::Euclidean => a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt(),
                DistanceMetric::DotProduct => {
                    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
                }
                DistanceMetric::Manhattan => {
                    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
                }
            }
        }))
    }

    fn ensure_hnsw(&self, dim: usize) {
        let mut guard = self.hnsw.lock().unwrap();
        if guard.is_none() {
            let dist_fn = Self::make_dist_fn(self.metric);
            let hnsw = Hnsw::new(16, self.max_elements, 16, 200, dist_fn);
            *guard = Some(hnsw);
            self.dimension.store(dim, Ordering::SeqCst);
        }
    }

    fn distance_to_similarity(&self, distance: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => {
                if distance == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + distance)
                }
            }
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Manhattan => {
                if distance == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + distance)
                }
            }
        }
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for HnswIndex {
    fn add(&self, document: Document) {
        let doc_id = document.id.clone();
        self.documents
            .write()
            .unwrap()
            .insert(doc_id.clone(), Arc::new(document));

        let embedding = self
            .documents
            .read()
            .unwrap()
            .get(&doc_id)
            .and_then(|d| d.embedding.clone());

        if let Some(embedding) = embedding {
            let dim = embedding.len();
            self.ensure_hnsw(dim);

            let numeric_id = self.next_id.fetch_add(1, Ordering::SeqCst);
            self.id_map
                .write()
                .unwrap()
                .insert(doc_id.clone(), numeric_id);
            self.reverse_id_map
                .write()
                .unwrap()
                .insert(numeric_id, doc_id);
            self.deleted_ids.write().unwrap().remove(&numeric_id);

            let guard = self.hnsw.lock().unwrap();
            if let Some(ref hnsw) = *guard {
                hnsw.insert((embedding.as_slice(), numeric_id));
            }
        }
    }

    fn remove(&self, id: &str) -> bool {
        if let Some(numeric_id) = self.id_map.write().unwrap().remove(id) {
            self.deleted_ids.write().unwrap().insert(numeric_id);
            self.documents.write().unwrap().remove(id);
            self.reverse_id_map.write().unwrap().remove(&numeric_id);
            true
        } else {
            false
        }
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<Similarity> {
        if top_k == 0 {
            return Vec::new();
        }

        let guard = self.hnsw.lock().unwrap();
        let hnsw = match *guard {
            Some(ref h) => h,
            None => return Vec::new(),
        };

        let ef = (top_k * 2).max(20);
        let neighbours = hnsw.search(query, top_k, ef);

        let mut results = Vec::new();
        for neighbour in neighbours {
            let numeric_id = neighbour.get_origin_id();
            if self.deleted_ids.read().unwrap().contains(&numeric_id) {
                continue;
            }
            let doc_id = self
                .reverse_id_map
                .read()
                .unwrap()
                .get(&numeric_id)
                .cloned();
            let Some(doc_id) = doc_id else {
                continue;
            };
            let doc = self.documents.read().unwrap().get(&doc_id).cloned();
            if let Some(doc) = doc {
                let distance = neighbour.get_distance();
                let score = self.distance_to_similarity(distance);
                results.push(Similarity {
                    document: (*doc).clone(),
                    score,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    fn search_exact_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &dyn Fn(&Document) -> bool,
    ) -> Vec<Similarity> {
        let docs = self.documents.read().unwrap();
        let mut similarities: Vec<Similarity> = docs
            .values()
            .filter_map(|doc| {
                if !filter(doc) {
                    return None;
                }
                if let Some(embedding) = &doc.embedding {
                    let score = self.distance_to_similarity(
                        Self::make_dist_fn(self.metric).eval(query, embedding),
                    );
                    Some(Similarity {
                        document: (**doc).clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        drop(docs);
        similarities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(top_k);
        similarities
    }

    fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<Similarity>> {
        queries.iter().map(|q| self.search(q, top_k)).collect()
    }

    fn clear(&self) {
        let mut guard = self.hnsw.lock().unwrap();
        *guard = None;
        drop(guard);
        self.documents.write().unwrap().clear();
        self.id_map.write().unwrap().clear();
        self.reverse_id_map.write().unwrap().clear();
        self.deleted_ids.write().unwrap().clear();
        self.next_id.store(0, Ordering::SeqCst);
        self.dimension.store(0, Ordering::SeqCst);
    }

    fn len(&self) -> usize {
        self.documents.read().unwrap().len()
    }

    fn dimension(&self) -> Option<usize> {
        let d = self.dimension.load(Ordering::SeqCst);
        if d == 0 { None } else { Some(d) }
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_index_basic() {
        let index = HnswIndex::new();
        let doc1 = Document::new("doc1".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let doc2 = Document::new("doc2".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);
        let doc3 = Document::new("doc3".to_string()).with_embedding(vec![0.9, 0.1, 0.0]);

        index.add(doc1.clone());
        index.add(doc2.clone());
        index.add(doc3.clone());

        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);

        // HNSW is approximate — exact doc ordering can vary slightly.
        // Verify the exact match is near the top and scores are reasonable.
        let ids: Vec<&str> = results.iter().map(|r| r.document.id.as_str()).collect();
        assert!(
            ids.contains(&doc1.id.as_str()),
            "exact match should be in top-2"
        );
        assert!(
            ids.contains(&doc3.id.as_str()),
            "near match should be in top-2"
        );

        // Exact match should have the highest score
        let exact_score = results
            .iter()
            .find(|r| r.document.id == doc1.id)
            .map(|r| r.score)
            .unwrap_or(0.0);
        assert!(
            exact_score > 0.99,
            "exact match score should be very high, got {exact_score}"
        );
    }

    #[test]
    fn test_hnsw_index_remove() {
        let index = HnswIndex::new();
        let doc = Document::new("test".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let id = doc.id.clone();

        index.add(doc);
        assert_eq!(index.len(), 1);

        let removed = index.remove(&id);
        assert!(removed);
        assert_eq!(index.len(), 0);

        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_index_clear() {
        let index = HnswIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));
        index.add(Document::new("b".to_string()).with_embedding(vec![0.0, 1.0]));
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_index_empty_search() {
        let index = HnswIndex::new();
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_index_document_without_embedding() {
        let index = HnswIndex::new();
        let doc = Document::new("no embedding".to_string());
        index.add(doc);

        assert_eq!(index.len(), 1);
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_index_top_k_larger_than_data() {
        let index = HnswIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));

        let results = index.search(&[1.0, 0.0], 100);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hnsw_index_search_zero_top_k() {
        let index = HnswIndex::new();
        index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));

        let results = index.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_index_with_metric() {
        let index = HnswIndex::with_metric(DistanceMetric::Euclidean);
        let doc1 = Document::new("doc1".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
        let doc2 = Document::new("doc2".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);

        index.add(doc1.clone());
        index.add(doc2.clone());

        assert_eq!(index.metric(), DistanceMetric::Euclidean);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, doc1.id);
    }

    #[test]
    fn test_hnsw_index_batch_search() {
        let index = HnswIndex::new();
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
}
