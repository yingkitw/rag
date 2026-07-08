//! Optional re-ranking hook for similarity lists (default: identity).

use std::future::Future;

use crate::errors::Result;
use crate::vector_store::Similarity;

pub trait SimilarityReranker: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        items: Vec<Similarity>,
    ) -> impl Future<Output = Result<Vec<Similarity>>> + Send;
}

/// No-op reranker (preserves order and scores).
pub struct PassthroughReranker;

impl SimilarityReranker for PassthroughReranker {
    async fn rerank(&self, _query: &str, items: Vec<Similarity>) -> Result<Vec<Similarity>> {
        Ok(items)
    }
}

/// Apply a reranker to an owned similarity list.
pub async fn rerank_similarities<R: SimilarityReranker>(
    reranker: &R,
    query: &str,
    items: Vec<Similarity>,
) -> Result<Vec<Similarity>> {
    reranker.rerank(query, items).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[tokio::test]
    async fn passthrough_preserves() {
        let r = PassthroughReranker;
        let v = vec![Similarity {
            document: Document::new("a".to_string()),
            score: 0.5,
        }];
        let out = r.rerank("q", v).await.unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].document.content, "a");
    }
}
