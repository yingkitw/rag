//! Combine vector similarity hits with BM25 keyword scores.

use std::collections::HashMap;

use crate::errors::{RagError, Result};
use crate::vector_store::{Document, Similarity};

fn min_max_norm(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &s in scores {
        min = min.min(s);
        max = max.max(s);
    }
    if (max - min).abs() < 1e-9 {
        return vec![0.5; scores.len()];
    }
    scores.iter().map(|s| (s - min) / (max - min)).collect()
}

/// Merge vector-ranked and BM25-ranked hits. `alpha` in [0, 1] weights the vector channel
/// (1 - alpha weights lexical). Documents are keyed by id via `docs`.
/// Fuse multiple ranked result lists using Reciprocal Rank Fusion.
/// `rank_constant` is the k parameter (default 60).
pub fn rrf_fusion(result_lists: &[Vec<Similarity>], rank_constant: usize, top_k: usize) -> Vec<Similarity> {
    if result_lists.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let mut scores: std::collections::HashMap<String, (f32, Option<Document>)> = std::collections::HashMap::new();
    for results in result_lists {
        for (rank, item) in results.iter().enumerate() {
            let rrf_score = 1.0 / (rank_constant as f32 + (rank + 1) as f32);
            let entry = scores.entry(item.document.id.clone()).or_insert((0.0, None));
            entry.0 += rrf_score;
            if entry.1.is_none() {
                entry.1 = Some(item.document.clone());
            }
        }
    }
    let mut fused: Vec<(String, f32, Document)> = scores
        .into_iter()
        .filter_map(|(id, (score, doc))| doc.map(|d| (id, score, d)))
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);
    fused.into_iter().map(|(_id, score, doc)| Similarity { document: doc, score }).collect()
}

pub fn merge_hybrid(
    docs_by_id: &HashMap<String, crate::vector_store::Document>,
    vector_hits: &[Similarity],
    keyword_hits: &[(String, f32)],
    alpha: f32,
    top_k: usize,
) -> Result<Vec<Similarity>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(RagError::InvalidConfig(format!(
            "merge_hybrid alpha must be within [0, 1], got {alpha}"
        )));
    }
    if top_k == 0 {
        return Ok(Vec::new());
    }

    let vector_scores: std::collections::HashMap<String, f32> = vector_hits
        .iter()
        .map(|s| (s.document.id.clone(), s.score))
        .collect();
    let keyword_scores: std::collections::HashMap<String, f32> = keyword_hits
        .iter()
        .map(|(id, score)| (id.clone(), *score))
        .collect();

    let ids: Vec<String> = vector_scores
        .keys()
        .chain(keyword_scores.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if ids.is_empty() {
        return Ok(Vec::new());
    }

    let v_raw: Vec<f32> = ids
        .iter()
        .map(|id| *vector_scores.get(id).unwrap_or(&0.0))
        .collect();
    let k_raw: Vec<f32> = ids
        .iter()
        .map(|id| *keyword_scores.get(id).unwrap_or(&0.0))
        .collect();

    let v_n = min_max_norm(&v_raw);
    let k_n = min_max_norm(&k_raw);

    let mut combined: Vec<(String, f32)> = ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            let score = alpha * v_n[i] + (1.0 - alpha) * k_n[i];
            (id, score)
        })
        .collect();

    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::new();
    for (id, score) in combined.into_iter().take(top_k) {
        let Some(doc) = docs_by_id.get(&id).cloned() else {
            continue;
        };
        out.push(Similarity { document: doc, score });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[test]
    fn rrf_fuses_by_rank() {
        let d1 = Document::new("a".to_string());
        let d2 = Document::new("b".to_string());
        let d3 = Document::new("c".to_string());
        let list1 = vec![
            Similarity { document: d1.clone(), score: 1.0 },
            Similarity { document: d2.clone(), score: 0.5 },
        ];
        let list2 = vec![
            Similarity { document: d3.clone(), score: 1.0 },
            Similarity { document: d1.clone(), score: 0.5 },
        ];
        let fused = rrf_fusion(&[list1, list2], 60, 10);
        assert!(!fused.is_empty());
        assert!(fused.iter().any(|s| s.document.id == d1.id));
    }

    #[test]
    fn merge_balances_channels() {
        let d1 = Document::new("rust systems".to_string());
        let d2 = Document::new("python scripts".to_string());
        let mut m = HashMap::new();
        m.insert(d1.id.clone(), d1.clone());
        m.insert(d2.id.clone(), d2.clone());

        let vec_hits = vec![
            Similarity {
                document: d1.clone(),
                score: 1.0,
            },
            Similarity {
                document: d2.clone(),
                score: 0.2,
            },
        ];
        let kw = vec![(d2.id.clone(), 5.0_f32), (d1.id.clone(), 0.1_f32)];
        let merged = merge_hybrid(&m, &vec_hits, &kw, 0.5, 2).unwrap();
        assert_eq!(merged.len(), 2);
    }
}
