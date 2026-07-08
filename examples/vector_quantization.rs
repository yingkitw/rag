//! Example: Int8 vector quantization for ~4x memory reduction.

use rag::quantize::{QuantizationParams, QuantizedIndex};

fn main() {
    let vectors = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.9, 0.8, 0.7, 0.6],
        vec![0.1, 0.9, 0.1, 0.9],
    ];

    // Fit per-dimension min/max bounds from the data.
    let params = QuantizationParams::fit(&vectors).unwrap();

    let mut index = QuantizedIndex::new(params);
    index.add("a", &[0.1, 0.2, 0.3, 0.4]);
    index.add("b", &[0.9, 0.8, 0.7, 0.6]);
    index.add("c", &[0.1, 0.9, 0.1, 0.9]);

    println!("Quantized index size: {} vectors", index.len());

    let results = index.search(&[0.9, 0.8, 0.7, 0.6], 3);
    println!("Approx top-k (query ~ vector b):");
    for (id, score) in &results {
        println!("  {id}  score={score:.4}");
    }
}
