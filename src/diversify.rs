use crate::vector_store::Similarity;
use std::collections::HashMap;

/// Cap the number of results sharing the same metadata attribute value.
/// `attribute` is the metadata key to diversify on; `max_per` is the cap per value.
pub fn diversify(results: Vec<Similarity>, attribute: &str, max_per: usize, total: usize) -> Vec<Similarity> {
    if max_per == 0 || total == 0 {
        return Vec::new();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for s in results {
        let key = s.document.metadata.get(attribute).cloned().unwrap_or_else(|| "__none__".to_string());
        let count = counts.entry(key.clone()).or_insert(0);
        if *count < max_per {
            *count += 1;
            out.push(s);
        }
        if out.len() >= total {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[test]
    fn diversify_caps_per_source() {
        let mut docs = Vec::new();
        for i in 0..5 {
            let mut doc = Document::new(format!("doc {}", i));
            doc.metadata.insert("source".to_string(), "A".to_string());
            docs.push(Similarity { document: doc, score: 1.0 - i as f32 * 0.1 });
        }
        let out = diversify(docs, "source", 2, 10);
        assert_eq!(out.len(), 2);
    }
}
