//! Sparse vector representation and search (SPLADE-style lexical-semantic signals).

use std::collections::HashMap;

/// Sparse vector as a map of dimension -> weight.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseVector {
    pub dims: HashMap<usize, f32>,
}

impl SparseVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(mut self, dim: usize, weight: f32) -> Self {
        self.dims.insert(dim, weight);
        self
    }

    /// Dot product with another sparse vector.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut sum = 0.0;
        for (dim, w1) in &self.dims {
            if let Some(w2) = other.dims.get(dim) {
                sum += w1 * w2;
            }
        }
        sum
    }

    /// Sparse cosine similarity.
    pub fn cosine(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let norm1 = self.norm();
        let norm2 = other.norm();
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        dot / (norm1 * norm2)
    }

    fn norm(&self) -> f32 {
        self.dims.values().map(|v| v * v).sum::<f32>().sqrt()
    }
}

/// Index for sparse vectors.
#[derive(Default)]
pub struct SparseIndex {
    vectors: Vec<(String, SparseVector)>,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: String, vector: SparseVector) {
        self.vectors.push((id, vector));
    }

    pub fn search(&self, query: &SparseVector, top_k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| (id.clone(), query.cosine(vec)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_dot_product() {
        let a = SparseVector::new().insert(0, 1.0).insert(2, 2.0);
        let b = SparseVector::new().insert(2, 3.0).insert(5, 1.0);
        assert_eq!(a.dot(&b), 6.0);
    }

    #[test]
    fn sparse_index_search() {
        let mut idx = SparseIndex::new();
        idx.add("a".to_string(), SparseVector::new().insert(0, 1.0));
        idx.add("b".to_string(), SparseVector::new().insert(1, 1.0));
        let results = idx.search(&SparseVector::new().insert(0, 1.0), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }
}
