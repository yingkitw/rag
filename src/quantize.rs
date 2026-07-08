//! Scalar Int8 vector quantization for ~4x memory reduction vs f32.
//!
//! Each vector is quantized per-dimension within the bounds of a *calibration*
//! set (min/max per dimension derived from a sample of vectors). Dequantized
//! dot products approximate the originals and are suitable for an initial ANN
//! coarse pass before an exact rerank.

use std::collections::HashMap;

/// Per-dimension quantization calibration: lower and upper bounds.
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub dim: usize,
}

impl QuantizationParams {
    /// Fit calibration bounds from a sample of vectors.
    /// If a dimension is constant (min == max) a tiny epsilon range is used so
    /// the value quantizes to a stable code rather than dividing by zero.
    pub fn fit(vectors: &[Vec<f32>]) -> crate::errors::Result<Self> {
        if vectors.is_empty() {
            return Err(crate::errors::RagError::InvalidConfig(
                "Cannot fit quantization on empty vector set".to_string(),
            ));
        }
        let dim = vectors[0].len();
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            if v.len() != dim {
                return Err(crate::errors::RagError::InvalidConfig(format!(
                    "Vector dimension mismatch: expected {dim}, got {}",
                    v.len()
                )));
            }
            for (i, x) in v.iter().enumerate() {
                if *x < min[i] {
                    min[i] = *x;
                }
                if *x > max[i] {
                    max[i] = *x;
                }
            }
        }
        for i in 0..dim {
            if max[i] <= min[i] {
                let eps = 1e-6_f32;
                let mid = min[i];
                min[i] = mid - eps;
                max[i] = mid + eps;
            }
        }
        Ok(Self { min, max, dim })
    }

    /// Quantize a single f32 vector into Int8 codes.
    pub fn quantize(&self, vector: &[f32]) -> Vec<i8> {
        let mut out = vec![0i8; self.dim];
        for i in 0..self.dim {
            let v = vector.get(i).copied().unwrap_or(self.min[i]);
            let range = self.max[i] - self.min[i];
            // Map [min, max] -> [-128, 127]
            let normalized = ((v - self.min[i]) / range).clamp(0.0, 1.0);
            out[i] = (normalized * 255.0 - 128.0).round() as i8;
        }
        out
    }

    /// Dequantize an Int8 code back to an approximate f32 vector.
    pub fn dequantize(&self, codes: &[i8]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.dim];
        for i in 0..self.dim {
            let c = codes.get(i).copied().unwrap_or(0) as f32;
            let range = self.max[i] - self.min[i];
            out[i] = self.min[i] + ((c + 128.0) / 255.0) * range;
        }
        out
    }
}

/// A quantized vector store. Lookup is by id; search returns top-k by
/// approximate dot product (dequantized), preserving the order of insertion.
#[derive(Debug, Clone)]
pub struct QuantizedIndex {
    pub params: QuantizationParams,
    codes: HashMap<String, Vec<i8>>,
}

impl QuantizedIndex {
    pub fn new(params: QuantizationParams) -> Self {
        Self {
            params,
            codes: HashMap::new(),
        }
    }

    pub fn add(&mut self, id: impl Into<String>, vector: &[f32]) {
        let q = self.params.quantize(vector);
        self.codes.insert(id.into(), q);
    }

    pub fn remove(&mut self, id: &str) -> bool {
        self.codes.remove(id).is_some()
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Approximate top-k by dot product. The query is dequantized lazily per
    /// candidate; this is intentionally simple — the win is storage, not speed.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let q = self.params.quantize(query);
        let qd = self.params.dequantize(&q);
        let mut scored: Vec<(String, f32)> = self
            .codes
            .iter()
            .map(|(id, codes)| {
                let dq = self.params.dequantize(codes);
                let dot: f32 = dq.iter().zip(qd.iter()).map(|(a, b)| a * b).sum();
                (id.clone(), dot)
            })
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
    fn test_fit_and_quantize_roundtrip() {
        let vectors = vec![
            vec![-1.0, 0.0, 1.0],
            vec![0.5, -0.5, 0.5],
            vec![2.0, 1.0, -1.0],
        ];
        let params = QuantizationParams::fit(&vectors).unwrap();
        let q = params.quantize(&[0.0, 0.0, 0.0]);
        assert_eq!(q.len(), 3);
        let dq = params.dequantize(&q);
        // Dequantized value should be close to the original.
        for (orig, back) in [0.0_f32, 0.0, 0.0].iter().zip(dq.iter()) {
            assert!((orig - back).abs() < 0.05, "orig={orig} back={back}");
        }
    }

    #[test]
    fn test_constant_dimension() {
        let vectors = vec![vec![5.0, 1.0], vec![5.0, 2.0]];
        let params = QuantizationParams::fit(&vectors).unwrap();
        let q = params.quantize(&[5.0, 1.5]);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_fit_empty_errors() {
        assert!(QuantizationParams::fit(&[]).is_err());
    }

    #[test]
    fn test_fit_dim_mismatch_errors() {
        let vectors = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(QuantizationParams::fit(&vectors).is_err());
    }

    #[test]
    fn test_quantized_index_search_order() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let params = QuantizationParams::fit(&vectors).unwrap();
        let mut index = QuantizedIndex::new(params);
        index.add("a", &[1.0, 0.0]);
        index.add("b", &[0.0, 1.0]);
        index.add("c", &[1.0, 1.0]);
        let res = index.search(&[1.0, 1.0], 3);
        assert_eq!(res.len(), 3);
        // Most aligned with query should be "c".
        assert_eq!(res[0].0, "c");
    }

    #[test]
    fn test_quantized_index_remove() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let params = QuantizationParams::fit(&vectors).unwrap();
        let mut index = QuantizedIndex::new(params);
        index.add("a", &[1.0, 0.0]);
        assert_eq!(index.len(), 1);
        assert!(index.remove("a"));
        assert!(index.is_empty());
        assert!(!index.remove("a"));
    }

    #[test]
    fn test_quantized_index_empty_search() {
        let params = QuantizationParams::fit(&[vec![1.0, 0.0]]).unwrap();
        let index = QuantizedIndex::new(params);
        let res = index.search(&[1.0, 0.0], 5);
        assert!(res.is_empty());
    }
}
