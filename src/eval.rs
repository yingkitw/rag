//! Retrieval-quality evaluation metrics: Recall@k, Precision@k, MRR, MAP, NDCG.
//!
//! These are pure functions over ranked retrieved id lists and sets of relevant
//! ids. They are deliberately model-agnostic so they can score any retriever
//! (vector, hybrid, graph) against a labelled ground-truth dataset.

use std::collections::HashSet;
use std::hash::Hash;

/// Convert an iterable of relevant ids into a `HashSet`.
pub fn relevance_set<T, I>(relevant: I) -> HashSet<T>
where
    T: Eq + Hash + Clone,
    I: IntoIterator<Item = T>,
{
    relevant.into_iter().collect()
}

/// Recall@k — fraction of relevant items retrieved in the top-k.
/// `retrieved` is assumed ordered best-first.
pub fn recall_at_k<T>(retrieved: &[T], relevant: &HashSet<T>, k: usize) -> f64
where
    T: Eq + Hash,
{
    if relevant.is_empty() {
        return 1.0;
    }
    let k = k.min(retrieved.len());
    let hits = retrieved[..k].iter().filter(|id| relevant.contains(id)).count();
    hits as f64 / relevant.len() as f64
}

/// Precision@k — fraction of retrieved top-k items that are relevant.
pub fn precision_at_k<T>(retrieved: &[T], relevant: &HashSet<T>, k: usize) -> f64
where
    T: Eq + Hash,
{
    if k == 0 {
        return 0.0;
    }
    let k = k.min(retrieved.len());
    if k == 0 {
        return 0.0;
    }
    let hits = retrieved[..k].iter().filter(|id| relevant.contains(id)).count();
    hits as f64 / k as f64
}

/// Mean Reciprocal Rank — reciprocal of the rank of the first relevant hit.
pub fn reciprocal_rank<T>(retrieved: &[T], relevant: &HashSet<T>) -> f64
where
    T: Eq + Hash,
{
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Average Precision for a single query — precision averaged over relevant ranks.
pub fn average_precision<T>(retrieved: &[T], relevant: &HashSet<T>) -> f64
where
    T: Eq + Hash,
{
    if relevant.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum_precisions = 0.0;
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum_precisions += hits as f64 / (i as f64 + 1.0);
        }
    }
    sum_precisions / relevant.len() as f64
}

/// Discounted Cumulative Gain at k. `relevance` is ordered best-first and
/// carries a graded relevance score per retrieved item (0 = irrelevant).
pub fn dcg_at_k(relevance: &[f64], k: usize) -> f64 {
    relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, rel)| {
            let denom = (i as f64 + 2.0).log2();
            if denom == 0.0 {
                0.0
            } else {
                (2f64.powf(*rel) - 1.0) / denom
            }
        })
        .sum()
}

/// Ideal DCG at k given the full sorted graded-relevance scores.
pub fn idcg_at_k(mut ideal_relevance: Vec<f64>, k: usize) -> f64 {
    ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    dcg_at_k(&ideal_relevance, k)
}

/// Normalized DCG at k. `relevance` is graded relevance for each retrieved
/// item (ordered best-first); `ideal_relevance` is the full relevance pool
/// used to compute the ideal ranking.
pub fn ndcg_at_k(relevance: &[f64], ideal_relevance: &[f64], k: usize) -> f64 {
    let ideal = idcg_at_k(ideal_relevance.to_vec(), k);
    if ideal == 0.0 {
        return 0.0;
    }
    dcg_at_k(relevance, k) / ideal
}

/// Build a graded-relevance vector for retrieved ids given a relevance lookup
/// (id -> graded score). Missing ids score 0.
pub fn graded_relevance<T>(retrieved: &[T], lookup: &std::collections::HashMap<T, f64>) -> Vec<f64>
where
    T: Eq + Hash + Clone,
{
    retrieved
        .iter()
        .map(|id| *lookup.get(id).unwrap_or(&0.0))
        .collect()
}

/// Aggregated metrics for a set of queries. Use this to average across a
/// benchmark after computing per-query metrics.
#[derive(Debug, Clone, Default)]
pub struct EvalReport {
    pub n_queries: usize,
    pub recall: f64,
    pub precision: f64,
    pub mrr: f64,
    pub map: f64,
    pub ndcg: f64,
}

impl EvalReport {
    /// Average a slice of per-query scalars into a single number.
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
}

/// Evaluate a retriever over multiple (query, relevant_ids) pairs at a fixed k.
///
/// `retrieve` maps a query string to its ranked list of document ids.
pub async fn evaluate<F, Fut, T, S>(
    queries: Vec<(S, Vec<T>)>,
    k: usize,
    retrieve: F,
) -> EvalReport
where
    T: Eq + Hash + Clone + Send + 'static,
    S: AsRef<str> + Send + 'static,
    F: Fn(String) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Vec<T>> + Send,
{
    let n = queries.len();
    if n == 0 {
        return EvalReport::default();
    }
    let mut recalls = Vec::with_capacity(n);
    let mut precisions = Vec::with_capacity(n);
    let mut rrs = Vec::with_capacity(n);
    let mut aps = Vec::with_capacity(n);
    let mut ndcgs = Vec::with_capacity(n);

    for (query, relevant) in queries {
        let retrieved = retrieve(query.as_ref().to_string()).await;
        let rel_set: HashSet<T> = relevant.iter().cloned().collect();
        recalls.push(recall_at_k(&retrieved, &rel_set, k));
        precisions.push(precision_at_k(&retrieved, &rel_set, k));
        rrs.push(reciprocal_rank(&retrieved, &rel_set));
        aps.push(average_precision(&retrieved, &rel_set));
        // Binary relevance as graded for NDCG when no grades supplied.
        let grades: std::collections::HashMap<T, f64> =
            relevant.iter().map(|id| (id.clone(), 1.0)).collect();
        let rel_vec = graded_relevance(&retrieved, &grades);
        let ideal: Vec<f64> = relevant.iter().map(|_| 1.0).collect();
        ndcgs.push(ndcg_at_k(&rel_vec, &ideal, k));
    }

    EvalReport {
        n_queries: n,
        recall: EvalReport::mean(&recalls),
        precision: EvalReport::mean(&precisions),
        mrr: EvalReport::mean(&rrs),
        map: EvalReport::mean(&aps),
        ndcg: EvalReport::mean(&ndcgs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec!["a", "b", "c", "d"];
        let relevant: HashSet<&str> = ["b", "d", "z"].iter().copied().collect();
        assert!((recall_at_k(&retrieved, &relevant, 4) - 2.0 / 3.0).abs() < 1e-9);
        assert!((recall_at_k(&retrieved, &relevant, 2) - 1.0 / 3.0).abs() < 1e-9);
        assert!((recall_at_k(&retrieved, &relevant, 0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_empty_relevant() {
        let retrieved = vec!["a", "b"];
        let relevant: HashSet<&str> = [].into_iter().collect();
        assert_eq!(recall_at_k(&retrieved, &relevant, 2), 1.0);
    }

    #[test]
    fn test_precision_at_k() {
        let retrieved = vec!["a", "b", "c", "d"];
        let relevant: HashSet<&str> = ["b"].iter().copied().collect();
        assert!((precision_at_k(&retrieved, &relevant, 2) - 0.5).abs() < 1e-9);
        assert!((precision_at_k(&retrieved, &relevant, 4) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_reciprocal_rank() {
        let relevant: HashSet<&str> = ["c"].iter().copied().collect();
        assert_eq!(reciprocal_rank(&["a", "b", "c"], &relevant), 1.0 / 3.0);
        let relevant2: HashSet<&str> = ["z"].iter().copied().collect();
        assert_eq!(reciprocal_rank(&["a", "b", "c"], &relevant2), 0.0);
    }

    #[test]
    fn test_average_precision() {
        let retrieved = vec!["a", "b", "c", "d"];
        let relevant: HashSet<&str> = ["a", "c"].iter().copied().collect();
        // AP = (1/1 + 2/3) / 2
        let expected = (1.0 + 2.0 / 3.0) / 2.0;
        assert!((average_precision(&retrieved, &relevant) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_dcg_and_ndcg() {
        // relevance grades for retrieved ranking
        let rel = vec![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let ideal = vec![3.0, 3.0, 2.0, 2.0, 1.0, 0.0];
        let ndcg = ndcg_at_k(&rel, &ideal, 6);
        assert!(ndcg > 0.0 && ndcg <= 1.0 + 1e-9);
        // Perfect ranking -> NDCG = 1
        let perfect = vec![3.0, 3.0, 2.0, 2.0, 1.0, 0.0];
        assert!((ndcg_at_k(&perfect, &perfect, 6) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_graded_relevance() {
        let mut lookup = std::collections::HashMap::new();
        lookup.insert("a", 2.0);
        lookup.insert("c", 3.0);
        let grades = graded_relevance(&["a", "b", "c"], &lookup);
        assert_eq!(grades, vec![2.0, 0.0, 3.0]);
    }

    #[tokio::test]
    async fn test_evaluate_aggregates() {
        let queries = vec![
            ("q1".to_string(), vec!["a", "b"]),
            ("q2".to_string(), vec!["c"]),
        ];
        let report = evaluate(queries, 2, |q| async move {
            if q == "q1" {
                vec!["a", "x"]
            } else {
                vec!["c", "y"]
            }
        })
        .await;
        assert_eq!(report.n_queries, 2);
        // q1: 1/2 relevant in top-2 = 0.5; q2: 1/1 = 1.0 -> avg 0.75
        assert!((report.recall - 0.75).abs() < 1e-9);
        // First hit is relevant at rank 1 for both queries.
        assert!((report.mrr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_relevance_set() {
        let set: HashSet<i32> = relevance_set(vec![1, 2, 2, 3]);
        assert_eq!(set.len(), 3);
    }
}
