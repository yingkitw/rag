//! Okapi BM25-style lexical scoring over in-memory documents.

use std::collections::HashMap;

use crate::vector_store::Document;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Tokenize for keyword search: lowercase alphanumeric terms.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            cur.push(c.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// BM25 index built from a set of documents (id + body).
pub struct Bm25Index {
    /// doc_id -> term frequencies
    tf: HashMap<String, HashMap<String, u32>>,
    /// term -> document frequency
    df: HashMap<String, u32>,
    dl: HashMap<String, u32>,
    n: usize,
    avgdl: f32,
}

impl Bm25Index {
    pub fn from_documents(docs: &[Document]) -> crate::errors::Result<Self> {
        if docs.is_empty() {
            return Ok(Self {
                tf: HashMap::new(),
                df: HashMap::new(),
                dl: HashMap::new(),
                n: 0,
                avgdl: 1.0,
            });
        }
        let mut tf = HashMap::new();
        let mut df = HashMap::new();
        let mut dl = HashMap::new();
        let mut total_len = 0u64;

        for doc in docs {
            let terms = tokenize(&doc.content);
            let len = terms.len() as u32;
            total_len += len as u64;
            dl.insert(doc.id.clone(), len);

            let mut freqs: HashMap<String, u32> = HashMap::new();
            let mut seen = std::collections::HashSet::new();
            for t in terms {
                *freqs.entry(t.clone()).or_insert(0) += 1;
                seen.insert(t);
            }
            for t in seen {
                *df.entry(t).or_insert(0) += 1;
            }
            tf.insert(doc.id.clone(), freqs);
        }

        let n = docs.len();
        let avgdl = total_len as f32 / n as f32;

        Ok(Self {
            tf,
            df,
            dl,
            n,
            avgdl,
        })
    }

    fn idf(&self, term: &str) -> f32 {
        let df = *self.df.get(term).unwrap_or(&0) as f32;
        if df <= 0.0 {
            return 0.0;
        }
        let n = self.n as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    fn score_doc(&self, doc_id: &str, query_terms: &[String]) -> f32 {
        let dl = *self.dl.get(doc_id).unwrap_or(&0) as f32;
        let Some(tf_map) = self.tf.get(doc_id) else {
            return 0.0;
        };

        let mut s = 0.0_f32;
        for t in query_terms {
            let f = *tf_map.get(t).unwrap_or(&0) as f32;
            if f <= 0.0 {
                continue;
            }
            let idf = self.idf(t);
            let num = f * (K1 + 1.0);
            let den = f + K1 * (1.0 - B + B * (dl / self.avgdl.max(1.0)));
            s += idf * (num / den);
        }
        s
    }

    /// Top-k documents by BM25 score for `query` text.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let terms = tokenize(query);
        if terms.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(String, f32)> = self
            .tf
            .keys()
            .map(|id| {
                let sc = self.score_doc(id, &terms);
                (id.clone(), sc)
            })
            .filter(|(_, sc)| *sc > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[test]
    fn bm25_prefers_matching_term() {
        let docs = vec![
            Document::new("alpha beta gamma".to_string()),
            Document::new("delta epsilon zeta".to_string()),
            Document::new("alpha omega alpha".to_string()),
        ];
        let idx = Bm25Index::from_documents(&docs).unwrap();
        let hits = idx.search("alpha", 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, docs[2].id);
    }
}
