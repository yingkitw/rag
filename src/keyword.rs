//! Okapi BM25-style lexical scoring over in-memory documents.

use std::collections::HashMap;

use crate::vector_store::Document;

/// Configurable BM25 hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Config {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Config {
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }
}

/// Tokenize for keyword search: lowercase alphanumeric terms.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(text.len() / 4);
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
    tf: HashMap<String, HashMap<String, u32>>,
    df: HashMap<String, u32>,
    dl: HashMap<String, u32>,
    n: usize,
    avgdl: f32,
    config: Bm25Config,
    /// doc_id -> raw content for phrase search
    content: HashMap<String, String>,
    /// term -> doc_ids containing the term; speeds up search by skipping non-matching docs
    postings: HashMap<String, Vec<String>>,
}

impl Bm25Index {
    pub fn from_documents(docs: &[Document]) -> crate::errors::Result<Self> {
        Self::from_documents_with_config(docs, Bm25Config::default())
    }

    pub fn from_documents_with_config(docs: &[Document], config: Bm25Config) -> crate::errors::Result<Self> {
        if docs.is_empty() {
            return Ok(Self {
                tf: HashMap::new(),
                df: HashMap::new(),
                dl: HashMap::new(),
                n: 0,
                avgdl: 1.0,
                config,
                content: HashMap::new(),
                postings: HashMap::new(),
            });
        }
        let mut tf = HashMap::new();
        let mut df = HashMap::new();
        let mut dl = HashMap::new();
        let mut content = HashMap::new();
        let mut total_len = 0u64;

        for doc in docs {
            let terms = tokenize(&doc.content);
            let len = terms.len() as u32;
            total_len += len as u64;
            dl.insert(doc.id.clone(), len);
            content.insert(doc.id.clone(), doc.content.to_lowercase());

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

        let mut postings: HashMap<String, Vec<String>> = HashMap::new();
        for (doc_id, freqs) in &tf {
            for term in freqs.keys() {
                postings.entry(term.clone()).or_default().push(doc_id.clone());
            }
        }

        Ok(Self {
            tf,
            df,
            dl,
            n,
            avgdl,
            config,
            content,
            postings,
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
        let k1 = self.config.k1;
        let b = self.config.b;

        let mut s = 0.0_f32;
        for t in query_terms {
            let f = *tf_map.get(t).unwrap_or(&0) as f32;
            if f <= 0.0 {
                continue;
            }
            let idf = self.idf(t);
            let num = f * (k1 + 1.0);
            let den = f + k1 * (1.0 - b + b * (dl / self.avgdl.max(1.0)));
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

        let mut candidates: std::collections::HashSet<&String> = std::collections::HashSet::new();
        for t in &terms {
            if let Some(ids) = self.postings.get(t) {
                for id in ids {
                    candidates.insert(id);
                }
            }
        }

        let mut scored: Vec<(String, f32)> = candidates
            .into_iter()
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

    /// Search for documents containing the exact phrase (case-insensitive).
    pub fn search_phrase(&self, phrase: &str, top_k: usize) -> Vec<(String, f32)> {
        let needle = phrase.to_lowercase();
        let mut scored: Vec<(String, f32)> = self
            .content
            .iter()
            .filter_map(|(id, text)| {
                let count = text.matches(&needle).count() as f32;
                if count > 0.0 {
                    Some((id.clone(), count))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Prefix search: documents whose content starts with the given prefix.
    pub fn search_prefix(&self, prefix: &str, top_k: usize) -> Vec<(String, f32)> {
        let pre = prefix.to_lowercase();
        let mut scored: Vec<(String, f32)> = self
            .content
            .iter()
            .filter_map(|(id, text)| {
                if text.starts_with(&pre) {
                    Some((id.clone(), 1.0))
                } else {
                    None
                }
            })
            .collect();
        scored.truncate(top_k);
        scored
    }
}

/// Field-aware BM25 index. Each document can have multiple named text fields with boost weights.
pub struct FieldBm25Index {
    fields: Vec<(String, f32)>, // (field_name, boost_weight)
    indexes: HashMap<String, Bm25Index>,
}

impl FieldBm25Index {
    pub fn new(fields: Vec<(String, f32)>) -> Self {
        Self { fields, indexes: HashMap::new() }
    }

    pub fn build(&mut self, docs: &[Document]) -> crate::errors::Result<()> {
        for (field_name, _) in &self.fields {
            let field_docs: Vec<Document> = docs.iter().map(|doc| {
                let text = doc.metadata.get(field_name).cloned().unwrap_or_default();
                Document::with_id(doc.id.clone(), text)
            }).collect();
            let idx = Bm25Index::from_documents(&field_docs)?;
            self.indexes.insert(field_name.clone(), idx);
        }
        Ok(())
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut combined: HashMap<String, f32> = HashMap::new();
        for (field_name, weight) in &self.fields {
            if let Some(idx) = self.indexes.get(field_name) {
                for (id, score) in idx.search(query, top_k * 2) {
                    *combined.entry(id).or_insert(0.0) += score * weight;
                }
            }
        }
        let mut scored: Vec<(String, f32)> = combined.into_iter().collect();
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

    #[test]
    fn configurable_bm25_params() {
        let docs = vec![
            Document::new("alpha beta gamma".to_string()),
            Document::new("alpha omega alpha".to_string()),
        ];
        let idx = Bm25Index::from_documents_with_config(&docs, Bm25Config::new(1.5, 0.5)).unwrap();
        let hits = idx.search("alpha", 2);
        assert!(!hits.is_empty());
    }

    #[test]
    fn phrase_search_finds_exact() {
        let docs = vec![
            Document::new("hello world here".to_string()),
            Document::new("hello there world".to_string()),
        ];
        let idx = Bm25Index::from_documents(&docs).unwrap();
        let hits = idx.search_phrase("hello world", 2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, docs[0].id);
    }

    #[test]
    fn prefix_search_works() {
        let docs = vec![
            Document::new("hello world".to_string()),
            Document::new("goodbye world".to_string()),
        ];
        let idx = Bm25Index::from_documents(&docs).unwrap();
        let hits = idx.search_prefix("hello", 2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, docs[0].id);
    }

    #[test]
    fn field_bm25_boosts() {
        let mut d1 = Document::new("content1".to_string());
        d1.metadata.insert("title".to_string(), "rust programming".to_string());
        d1.metadata.insert("body".to_string(), "some body text".to_string());
        let mut d2 = Document::new("content2".to_string());
        d2.metadata.insert("title".to_string(), "other topic".to_string());
        d2.metadata.insert("body".to_string(), "rust programming details".to_string());
        let docs = vec![d1, d2];
        let mut idx = FieldBm25Index::new(vec![
            ("title".to_string(), 3.0),
            ("body".to_string(), 1.0),
        ]);
        idx.build(&docs).unwrap();
        let hits = idx.search("rust programming", 2);
        assert!(!hits.is_empty());
    }
}
