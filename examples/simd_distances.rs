//! Example: SIMD-accelerated distance kernels (auto-uses AVX2/FMA when present).

use rag::simd;

fn main() {
    let a: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01 - 2.5).collect();
    let b: Vec<f32> = (0..512).map(|i| (i as f32) * 0.02 - 1.0).collect();

    println!("dim = {}", a.len());
    println!("dot product        = {:.4}", simd::dot_product(&a, &b));
    println!("cosine similarity  = {:.4}", simd::cosine_similarity(&a, &b));
    println!("euclidean distance = {:.4}", simd::euclidean_distance(&a, &b));
    println!("manhattan distance = {:.4}", simd::manhattan_distance(&a, &b));

    // Benchmark sketch: scalar vs SIMD should match.
    let scalar_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    println!("scalar dot (check) = {:.4}", scalar_dot);
}
