//! Example: Retrieval-quality evaluation metrics (Recall@k, Precision@k,
//! MRR, MAP, NDCG).

use rag::eval::{
    average_precision, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, relevance_set,
};
use std::collections::HashSet;

fn main() {
    // Ground truth: the relevant doc ids for a query.
    let relevant: HashSet<String> =
        relevance_set(["d2".to_string(), "d5".to_string(), "d9".to_string()]);

    // Ranked output from some retriever (best first).
    let retrieved = vec![
        "d2".to_string(),
        "d1".to_string(),
        "d9".to_string(),
        "d3".to_string(),
        "d5".to_string(),
    ];

    let k = 3;
    println!("Relevant: {:?}", relevant);
    println!("Retrieved: {:?}", retrieved);
    println!("Recall@{}    = {:.3}", k, recall_at_k(&retrieved, &relevant, k));
    println!("Precision@{} = {:.3}", k, precision_at_k(&retrieved, &relevant, k));
    println!("MRR          = {:.3}", reciprocal_rank(&retrieved, &relevant));
    println!("MAP          = {:.3}", average_precision(&retrieved, &relevant));

    // Graded-relevance NDCG: retrieved items carry grades 0..3.
    let grades = vec![3.0, 0.0, 2.0, 0.0, 1.0];
    let ideal = vec![3.0, 2.0, 1.0];
    println!("NDCG@{}       = {:.3}", k, ndcg_at_k(&grades, &ideal, k));
}
