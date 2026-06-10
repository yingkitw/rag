//! Example: Sparse vector search for lexical-semantic signals.
//!
//! Sparse vectors represent documents as dimension->weight maps,
//! enabling efficient keyword-semantic hybrid retrieval without dense embeddings.

use rag::sparse::{SparseIndex, SparseVector};

fn main() {
    let mut index = SparseIndex::new();

    // Index documents as sparse vectors (dimension = term hash, weight = TF-like)
    // In practice these come from a sparse encoder like SPLADE.
    index.add(
        "doc1".to_string(),
        SparseVector::new()
            .insert(0, 1.0)  // term "rust"
            .insert(2, 0.5)  // term "programming"
            .insert(5, 0.3), // term "memory"
    );
    index.add(
        "doc2".to_string(),
        SparseVector::new()
            .insert(1, 1.0)  // term "python"
            .insert(2, 0.4)  // term "programming"
            .insert(3, 0.6), // term "scripting"
    );
    index.add(
        "doc3".to_string(),
        SparseVector::new()
            .insert(0, 0.8)  // term "rust"
            .insert(2, 0.7)  // term "programming"
            .insert(5, 0.5), // term "memory"
    );

    // Query: "rust programming"
    let query = SparseVector::new()
        .insert(0, 1.0)  // rust
        .insert(2, 1.0); // programming

    let results = index.search(&query, 3);

    println!("Sparse vector search for 'rust programming':");
    for (id, score) in &results {
        println!("  {}  cosine={:.4}", id, score);
    }

    // Verify dot product
    let a = SparseVector::new().insert(0, 2.0).insert(5, 3.0);
    let b = SparseVector::new().insert(5, 4.0).insert(10, 1.0);
    println!("\nDot product of overlapping sparse vectors: {} (expected 12.0)", a.dot(&b));
}
