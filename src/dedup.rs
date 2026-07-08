//! Near-duplicate suppression using word-set Jaccard similarity on chunk text.

use std::collections::HashSet;

use crate::vector_store::Similarity;

fn word_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn jaccard_from_sets(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Jaccard similarity between token bags (words).
pub fn content_jaccard(a: &str, b: &str) -> f32 {
    jaccard_from_sets(&word_set(a), &word_set(b))
}

/// Keep first occurrence; drop later items whose content is too similar to an already kept item.
pub fn dedup_similarities(items: Vec<Similarity>, min_jaccard: f32) -> Vec<Similarity> {
    if !(0.0..=1.0).contains(&min_jaccard) {
        return items;
    }
    let mut kept: Vec<(Similarity, HashSet<String>)> = Vec::new();
    for s in items {
        let words = word_set(&s.document.content);
        let dup = kept
            .iter()
            .any(|(_, kwords)| jaccard_from_sets(kwords, &words) >= min_jaccard);
        if !dup {
            kept.push((s, words));
        }
    }
    kept.into_iter().map(|(s, _)| s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[test]
    fn dedup_drops_near_duplicate() {
        let items = vec![
            Similarity {
                document: Document::new("hello world foo".to_string()),
                score: 1.0,
            },
            Similarity {
                document: Document::new("hello world bar".to_string()),
                score: 0.9,
            },
            Similarity {
                document: Document::new("totally different".to_string()),
                score: 0.5,
            },
        ];
        let out = dedup_similarities(items, 0.4);
        assert_eq!(out.len(), 2);
    }
}
