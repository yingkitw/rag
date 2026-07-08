//! SIMD-accelerated distance / similarity kernels.
//!
//! These use platform intrinsics (`std::arch::x86_64`) with **runtime CPU
//! feature detection** and fall back to scalar code when AVX2/FMA is absent.
//! On AVX2 targets the dot/cosine kernels process 8 f32 lanes per instruction,
//! which is the 3–4x speedup noted by competing Rust RAG projects. All
//! functions are `#[inline]` and written to be auto-vectorizable even in the
//! scalar path.

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use std::arch::x86_64::*;

/// Dot product of two equal-length f32 slices.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && a.len() >= 8 {
            unsafe {
                return dot_fma(a, b);
            }
        }
    }
    scalar_dot(a, b)
}

/// L2 (Euclidean) distance.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "euclidean_distance: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && a.len() >= 8 {
            unsafe {
                let sum_sq = squared_diff_fma(a, b);
                return sum_sq.sqrt();
            }
        }
    }
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    sum_sq.sqrt()
}

/// Cosine similarity in [-1, 1].
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let dot = dot_product(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// L2 norm (Euclidean length).
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt()
}

/// Manhattan (L1) distance.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "manhattan_distance: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    let n8 = a.len() / 8 * 8;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = horizontal_sum256(acc);
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn squared_diff_fma(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    let n8 = a.len() / 8 * 8;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
        i += 8;
    }
    let mut sum = horizontal_sum256(acc);
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum256(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi128, lo128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), sums);
    out[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_matches_scalar() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 0.05).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((dot_product(&a, &b) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_dot_product_small() {
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_matches_scalar() {
        let a: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..50).map(|i| (i + 1) as f32).collect();
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        assert!((euclidean_distance(&a, &b) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_large_matches_scalar() {
        let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02).collect();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected = dot / (na * nb);
        assert!((cosine_similarity(&a, &b) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_norm_zero_vector() {
        assert_eq!(norm(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_manhattan_distance() {
        assert!((manhattan_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 3.0]) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_unit() {
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }
}
