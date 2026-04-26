use crate::chunker::TextChunker;
use crate::embeddings::EmbeddingModel;
use crate::errors::Result;
use crate::vector_store::{Document, VectorStore};

pub struct Retriever<T, V>
where
    T: EmbeddingModel,
    V: VectorStore,
{
    embedding_model: T,
    vector_store: V,
    chunker: Box<dyn TextChunker>,
    top_k: usize,
}

impl<T, V> Retriever<T, V>
where
    T: EmbeddingModel,
    V: VectorStore,
{
    pub fn new(embedding_model: T, vector_store: V) -> Self {
        Self {
            embedding_model,
            vector_store,
            chunker: Box::new(crate::chunker::FixedSizeChunker::default()),
            top_k: 5,
        }
    }

    pub fn with_chunker(mut self, chunker: Box<dyn TextChunker>) -> Self {
        self.chunker = chunker;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub async fn add_document(&self, content: String) -> Result<String> {
        let chunks = self.chunker.chunk(&content)?;

        let chunk_embeddings = self
            .embedding_model
            .embed(chunks.clone())
            .await?;

        let mut doc_ids = Vec::new();

        for (chunk, embedding) in chunks.into_iter().zip(chunk_embeddings.into_iter()) {
            let doc = Document::new(chunk)
                .with_embedding(embedding)
                .with_metadata("source".to_string(), "document".to_string());
            let id = doc.id.clone();
            self.vector_store.add(doc).await?;
            doc_ids.push(id);
        }

        Ok(doc_ids.join(","))
    }

    pub async fn add_document_with_metadata(
        &self,
        content: String,
        metadata: Vec<(String, String)>,
    ) -> Result<String> {
        let chunks = self.chunker.chunk(&content)?;

        let chunk_embeddings = self
            .embedding_model
            .embed(chunks.clone())
            .await?;

        let mut doc_ids = Vec::new();

        for (chunk, embedding) in chunks.into_iter().zip(chunk_embeddings.into_iter()) {
            let mut doc = Document::new(chunk)
                .with_embedding(embedding);
            for (key, value) in metadata.clone() {
                doc = doc.with_metadata(key, value);
            }
            let id = doc.id.clone();
            self.vector_store.add(doc).await?;
            doc_ids.push(id);
        }

        Ok(doc_ids.join(","))
    }

    pub async fn retrieve(&self, query: &str) -> Result<Vec<String>> {
        let query_embedding = self.embedding_model.embed_single(query).await?;
        let similarities = self
            .vector_store
            .search(&query_embedding, self.top_k)
            .await?;

        Ok(similarities
            .into_iter()
            .map(|s| s.document.content)
            .collect())
    }

    pub async fn retrieve_with_scores(&self, query: &str) -> Result<Vec<(String, f32)>> {
        let query_embedding = self.embedding_model.embed_single(query).await?;
        let similarities = self
            .vector_store
            .search(&query_embedding, self.top_k)
            .await?;

        Ok(similarities
            .into_iter()
            .map(|s| (s.document.content, s.score))
            .collect())
    }

    pub async fn retrieve_filtered(
        &self,
        query: &str,
        metadata_filter: &str,
    ) -> Result<Vec<String>> {
        let query_embedding = self.embedding_model.embed_single(query).await?;
        let mut similarities = self
            .vector_store
            .search(&query_embedding, self.top_k * 2)
            .await?;

        similarities.retain(|s| {
            s.document
                .metadata
                .values()
                .any(|v| v.contains(metadata_filter))
        });

        similarities.truncate(self.top_k);

        Ok(similarities
            .into_iter()
            .map(|s| s.document.content)
            .collect())
    }

    pub fn vector_store(&self) -> &V {
        &self.vector_store
    }

    pub fn embedding_model(&self) -> &T {
        &self.embedding_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingModel;
    use crate::errors::Result;
    use crate::vector_store::InMemoryVectorStore;
    use async_trait::async_trait;

    /// Mock embedding model that returns deterministic embeddings.
    /// Each word gets a simple embedding based on character sums.
    #[derive(Clone)]
    struct MockEmbeddingModel;

    #[async_trait]
    impl EmbeddingModel for MockEmbeddingModel {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            let mut results = Vec::new();
            for text in texts {
                let embedding = text
                    .bytes()
                    .map(|b| b as f32 / 255.0)
                    .collect::<Vec<f32>>();
                results.push(embedding);
            }
            Ok(results)
        }
    }

    #[tokio::test]
    async fn test_retriever_add_and_retrieve() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(100, 0)))
            .with_top_k(3);

        let ids = retriever
            .add_document("hello world test content".to_string())
            .await
            .unwrap();
        assert!(!ids.is_empty());

        let count = retriever.vector_store().count().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_retriever_add_with_metadata() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(50, 0)))
            .with_top_k(5);

        let ids = retriever
            .add_document_with_metadata(
                "document with metadata".to_string(),
                vec![
                    ("author".to_string(), "test".to_string()),
                    ("tag".to_string(), "important".to_string()),
                ],
            )
            .await
            .unwrap();
        assert!(!ids.is_empty());

        let doc = retriever
            .vector_store()
            .get(&ids.split(',').next().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(doc.metadata.get("author"), Some(&"test".to_string()));
        assert_eq!(doc.metadata.get("tag"), Some(&"important".to_string()));
    }

    #[tokio::test]
    async fn test_retriever_with_top_k() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(5, 0)))
            .with_top_k(2);

        retriever
            .add_document("one two three four five six seven eight nine ten eleven twelve".to_string())
            .await
            .unwrap();

        let count = retriever.vector_store().count().await.unwrap();
        assert!(count >= 2);

        let results = retriever.retrieve("query").await.unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_retriever_chunker_error() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(5, 10)))
            .with_top_k(3);

        let result = retriever.add_document("some text".to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retriever_retrieve_with_scores() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(100, 0)))
            .with_top_k(5);

        retriever
            .add_document("test document for scoring".to_string())
            .await
            .unwrap();

        let results = retriever.retrieve_with_scores("test").await.unwrap();
        assert!(!results.is_empty());
        for (_, score) in &results {
            assert!(*score >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_retriever_filtered_retrieve() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(100, 0)))
            .with_top_k(5);

        retriever
            .add_document_with_metadata(
                "special tagged content".to_string(),
                vec![("tag".to_string(), "special".to_string())],
            )
            .await
            .unwrap();

        let results = retriever.retrieve_filtered("content", "special").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "special tagged content");
    }

    #[tokio::test]
    async fn test_retriever_accessor_methods() {
        let model = MockEmbeddingModel;
        let store = InMemoryVectorStore::new();
        let retriever = Retriever::new(model, store)
            .with_chunker(Box::new(crate::chunker::FixedSizeChunker::new(100, 0)))
            .with_top_k(10);

        let _ = retriever.vector_store();
        let _ = retriever.embedding_model();
    }
}