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