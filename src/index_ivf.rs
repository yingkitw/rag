//! Inverted-file (IVF) index: cluster vectors by nearest centroid, probe top clusters only.
//!
//! Exact within probed clusters; suitable as a stepping stone before full HNSW.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use dashmap::DashMap;

use crate::index::{DistanceMetric, Index};
use crate::vector_store::{Document, Similarity};

/// IVF index with brute-force scoring inside selected clusters.
pub struct IvfflatIndex {
    metric: DistanceMetric,
    dimension: RwLock<Option<usize>>,
    nlist: usize,
    nprobe: usize,
    centroids: RwLock<Vec<Vec<f32>>>,
    buckets: RwLock<Vec<Vec<String>>>,
    doc_cluster: DashMap<String, usize>,
    documents: DashMap<String, Arc<Document>>,
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
            doc_cluster: DashMap::new(),
            documents: DashMap::new(),
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
        let mut similarities: Vec<Similarity> = self
            .documents
            .iter()
            .filter_map(|entry| {
                let doc = entry.value();
                doc.embedding.as_ref().map(|emb| Similarity {
                    document: (**doc).clone(),
                    score: self.metric.similarity(query, emb),
                })
            })
            .collect();
        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }
}

impl Index for IvfflatIndex {
    fn add(&self, document: Document) {
        let id = document.id.clone();
        let arc = Arc::new(document);
        self.documents.insert(id.clone(), arc.clone());

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
            self.doc_cluster.insert(id, idx);
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
        self.doc_cluster.insert(id, j);
    }

    fn remove(&self, id: &str) -> bool {
        if let Some((_, _)) = self.documents.remove(id) {
            if let Some((_, cluster)) = self.doc_cluster.remove(id) {
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
        if top_k == 0 || self.documents.is_empty() {
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
        let mut seen = std::collections::HashSet::new();
        let mut cand_ids = Vec::new();
        for &pi in &probe {
            if let Some(bucket) = buckets.get(pi) {
                for id in bucket {
                    if seen.insert(id.clone()) {
                        cand_ids.push(id.clone());
                    }
                }
            }
        }
        drop(buckets);

        let mut similarities: Vec<Similarity> = cand_ids
            .into_iter()
            .filter_map(|cid| {
                let doc = self.documents.get(&cid)?;
                let emb = doc.embedding.as_ref()?;
                Some(Similarity {
                    document: (**doc.value()).clone(),
                    score: self.metric.similarity(query, emb),
                })
            })
            .collect();

        if similarities.len() < top_k {
            let extra = self.full_scan(query, top_k);
            for s in extra {
                if similarities.len() >= top_k {
                    break;
                }
                if !similarities.iter().any(|x| x.document.id == s.document.id) {
                    similarities.push(s);
                }
            }
        }

        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    fn clear(&self) {
        self.documents.clear();
        self.doc_cluster.clear();
        self.centroids.write().unwrap().clear();
        self.buckets.write().unwrap().clear();
        self.centroid_count.store(0, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        *self.dimension.write().unwrap() = None;
    }

    fn len(&self) -> usize {
        self.documents.len()
    }

    fn dimension(&self) -> Option<usize> {
        *self.dimension.read().unwrap()
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
