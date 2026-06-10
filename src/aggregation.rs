use crate::vector_store::Document;
use std::collections::HashMap;

/// Count documents per value of a metadata key.
pub fn count_by(documents: &[Document], key: &str) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for doc in documents {
        if let Some(v) = doc.metadata.get(key) {
            *out.entry(v.clone()).or_insert(0) += 1;
        }
    }
    out
}

/// Group documents by a metadata key.
pub fn group_by<'a>(documents: &'a [Document], key: &str) -> HashMap<String, Vec<&'a Document>> {
    let mut out: HashMap<String, Vec<&Document>> = HashMap::new();
    for doc in documents {
        if let Some(v) = doc.metadata.get(key) {
            out.entry(v.clone()).or_default().push(doc);
        }
    }
    out
}

/// Sum a numeric metadata value per group key.
pub fn sum_by(documents: &[Document], group_key: &str, sum_key: &str) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for doc in documents {
        if let (Some(gv), Some(sv)) = (doc.metadata.get(group_key), doc.metadata.get(sum_key)) {
            if let Ok(val) = sv.parse::<f64>() {
                *out.entry(gv.clone()).or_insert(0.0) += val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_by_groups() {
        let docs = vec![
            Document::new("a".to_string()).with_metadata("tag".to_string(), "x".to_string()),
            Document::new("b".to_string()).with_metadata("tag".to_string(), "x".to_string()),
            Document::new("c".to_string()).with_metadata("tag".to_string(), "y".to_string()),
        ];
        let counts = count_by(&docs, "tag");
        assert_eq!(counts.get("x"), Some(&2));
        assert_eq!(counts.get("y"), Some(&1));
    }
}
