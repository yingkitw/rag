//! Inverted-file (IVF) index: cluster vectors by nearest centroid, probe top clusters only.
//!
//! Exact within probed clusters; suitable as a stepping stone before full HNSW.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::index::{DistanceMetric, Index, flat_search_top_k, flat_search_top_k_slice};
use crate::vector_store::{Document, Similarity};

/// IVF index with brute-force scoring inside selected clusters.
pub struct IvfflatIndex {
    metric: DistanceMetric,
    dimension: RwLock<Option<usize>>,
    nlist: usize,
    nprobe: usize,
    centroids: RwLock<Vec<Vec<f32>>>,
    buckets: RwLock<Vec<Vec<String>>>,
    doc_cluster: RwLock<HashMap<String, usize>>,
    documents: RwLock<HashMap<String, Arc<Document>>>,
    centroid_count: AtomicUsize,
    ready: AtomicBool,
}

impl IvfflatIndex {
    pub fn new(nlist: usize, nprobe: usize) -> Self {
        let nlist = nlist.max(1);
        let nprobe = nprobe.clamp(1, nlist);
        Self {
            metric: DistanceMetric::default(),
            dimension: RwLock::new(None),
            nlist,
            nprobe,
            centroids: RwLock::new(Vec::new()),
            buckets: RwLock::new(Vec::new()),
            doc_cluster: RwLock::new(HashMap::new()),
            documents: RwLock::new(HashMap::new()),
            centroid_count: AtomicUsize::new(0),
            ready: AtomicBool::new(false),
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    fn nearest_centroid(&self, centroids: &[Vec<f32>], query: &[f32]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.metric.similarity(query, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn full_scan(&self, query: &[f32], top_k: usize) -> Vec<Similarity> {
        flat_search_top_k(&self.documents, query, self.metric, top_k, &|_| true)
    }
}

impl Index for IvfflatIndex {
    fn add(&self, document: Document) {
        let id = document.id.clone();
        let arc = Arc::new(document);
        self.documents
            .write()
            .unwrap()
            .insert(id.clone(), arc.clone());

        let Some(emb) = arc.embedding.as_ref() else {
            return;
        };

        {
            let mut d = self.dimension.write().unwrap();
            match *d {
                None => *d = Some(emb.len()),
                Some(existing) if existing != emb.len() => return,
                Some(_) => {}
            }
        }

        let c = self.centroid_count.load(Ordering::Acquire);
        if c < self.nlist {
            let idx = c;
            self.centroids.write().unwrap().push(emb.clone());
            self.buckets.write().unwrap().push(vec![id.clone()]);
            self.doc_cluster.write().unwrap().insert(id, idx);
            let new_c = self.centroid_count.fetch_add(1, Ordering::AcqRel) + 1;
            if new_c >= self.nlist {
                self.ready.store(true, Ordering::Release);
            }
            return;
        }

        let centroids = self.centroids.read().unwrap();
        let j = self.nearest_centroid(&centroids, emb);
        drop(centroids);
        self.buckets.write().unwrap()[j].push(id.clone());
        self.doc_cluster.write().unwrap().insert(id, j);
    }

    fn remove(&self, id: &str) -> bool {
        if self.documents.write().unwrap().remove(id).is_some() {
            if let Some(cluster) = self.doc_cluster.write().unwrap().remove(id) {
                let mut buckets = self.buckets.write().unwrap();
                if let Some(bucket) = buckets.get_mut(cluster) {
                    bucket.retain(|x| x != id);
                }
            }
            return true;
        }
        false
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<Similarity> {
        if top_k == 0 || self.documents.read().unwrap().is_empty() {
            return Vec::new();
        }
        if !self.ready.load(Ordering::Acquire) {
            return self.full_scan(query, top_k);
        }

        let centroids = self.centroids.read().unwrap();
        if centroids.is_empty() {
            return self.full_scan(query, top_k);
        }

        let mut order: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.metric.similarity(query, c)))
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let probe = order
            .into_iter()
            .take(self.nprobe)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        drop(centroids);

        let buckets = self.buckets.read().unwrap();
        let docs = self.documents.read().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut candidates: Vec<Arc<Document>> = Vec::new();
        for &pi in &probe {
            if let Some(bucket) = buckets.get(pi) {
                for id in bucket {
                    if seen.insert(id.clone())
                        && let Some(doc) = docs.get(id)
                    {
                        candidates.push(Arc::clone(doc));
                    }
                }
            }
        }
        drop(buckets);
        drop(docs);

        let mut similarities =
            flat_search_top_k_slice(&candidates, query, self.metric, top_k, &|_| true);

        if similarities.len() < top_k {
            let extra = flat_search_top_k(&self.documents, query, self.metric, top_k, &|doc| {
                !similarities.iter().any(|s| s.document.id == doc.id)
            });
            similarities.extend(extra.into_iter().take(top_k - similarities.len()));
        }

        similarities
    }

    fn clear(&self) {
        self.documents.write().unwrap().clear();
        self.doc_cluster.write().unwrap().clear();
        self.centroids.write().unwrap().clear();
        self.buckets.write().unwrap().clear();
        self.centroid_count.store(0, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        *self.dimension.write().unwrap() = None;
    }

    fn len(&self) -> usize {
        self.documents.read().unwrap().len()
    }

    fn dimension(&self) -> Option<usize> {
        *self.dimension.read().unwrap()
    }

    fn search_exact_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &dyn Fn(&Document) -> bool,
    ) -> Vec<Similarity> {
        flat_search_top_k(&self.documents, query, self.metric, top_k, filter)
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_falls_back_before_trained() {
        let ix = IvfflatIndex::new(4, 2);
        ix.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0, 0.0]));
        let r = ix.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(r.len(), 1);
    }
}
