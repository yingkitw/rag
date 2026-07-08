//! Local ONNX-backed models via `fastembed` (no network at inference time).
//!
//! Enable with the `fastembed` feature. On first use, models are downloaded to
//! a local cache and then run entirely on-device through the ONNX Runtime.
//!
//! - [`FastEmbedEmbeddingModel`] implements [`crate::embeddings::EmbeddingModel`]
//!   for text.
//! - [`FastEmbedReranker`] implements [`crate::rerank::SimilarityReranker`] with
//!   a local cross-encoder (e.g. `bge-reranker`).
//! - [`FastEmbedImageEmbeddingModel`] (behind the `image-embeddings` feature)
//!   produces CLIP-style image embeddings.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::embeddings::EmbeddingModel;
use crate::errors::{RagError, Result};
use crate::rerank::SimilarityReranker;
use crate::vector_store::Similarity;

/// Local text embedding model backed by `fastembed`.
#[cfg(feature = "fastembed")]
pub struct FastEmbedEmbeddingModel {
    model: Mutex<Option<Arc<fastembed::TextEmbedding>>>,
    options: fastembed::InitOptions,
}

#[cfg(feature = "fastembed")]
impl FastEmbedEmbeddingModel {
    /// Create a model with default settings (all-MiniLM-L6-v2).
    pub fn new() -> Self {
        Self::with_options(fastembed::InitOptions::default())
    }

    /// Create a model with a specific embedding model.
    pub fn with_model(model: fastembed::EmbeddingModel) -> Self {
        Self::with_options(fastembed::InitOptions::new(model))
    }

    /// Create a model from full `InitOptions` (cache dir, model, etc.).
    pub fn with_options(options: fastembed::InitOptions) -> Self {
        Self {
            model: Mutex::new(None),
            options,
        }
    }

    async fn get(&self) -> Result<Arc<fastembed::TextEmbedding>> {
        let mut guard = self.model.lock().await;
        if let Some(m) = guard.as_ref() {
            return Ok(Arc::clone(m));
        }
        let m = Arc::new(
            fastembed::TextEmbedding::try_new(self.options.clone())
                .map_err(|e| RagError::EmbeddingError(format!("fastembed init: {e}")))?,
        );
        *guard = Some(Arc::clone(&m));
        Ok(m)
    }
}

#[cfg(feature = "fastembed")]
impl Default for FastEmbedEmbeddingModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "fastembed")]
impl EmbeddingModel for FastEmbedEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let model = self.get().await?;
        tokio::task::spawn_blocking(move || model.embed(texts, None))
            .await
            .map_err(|e| RagError::EmbeddingError(format!("join error: {e}")))?
            .map_err(|e| RagError::EmbeddingError(format!("fastembed embed: {e}")))
    }
}

/// Local cross-encoder reranker backed by `fastembed` (e.g. `bge-reranker`).
#[cfg(feature = "fastembed")]
pub struct FastEmbedReranker {
    model: Mutex<Option<Arc<fastembed::TextRerank>>>,
    options: fastembed::RerankInitOptions,
}

#[cfg(feature = "fastembed")]
impl FastEmbedReranker {
    /// Default reranker model (bge-reranker-base).
    pub fn new() -> Self {
        Self::with_model(fastembed::RerankerModel::BGERerankerBase)
    }

    /// Create a reranker with a specific model.
    pub fn with_model(model: fastembed::RerankerModel) -> Self {
        Self::with_options(fastembed::RerankInitOptions::new(model))
    }

    /// Create a reranker from full `RerankInitOptions`.
    pub fn with_options(options: fastembed::RerankInitOptions) -> Self {
        Self {
            model: Mutex::new(None),
            options,
        }
    }

    async fn get(&self) -> Result<Arc<fastembed::TextRerank>> {
        let mut guard = self.model.lock().await;
        if let Some(m) = guard.as_ref() {
            return Ok(Arc::clone(m));
        }
        let m = Arc::new(
            fastembed::TextRerank::try_new(self.options.clone())
                .map_err(|e| RagError::EmbeddingError(format!("fastembed reranker init: {e}")))?,
        );
        *guard = Some(Arc::clone(&m));
        Ok(m)
    }
}

#[cfg(feature = "fastembed")]
impl Default for FastEmbedReranker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "fastembed")]
impl SimilarityReranker for FastEmbedReranker {
    async fn rerank(&self, query: &str, items: Vec<Similarity>) -> Result<Vec<Similarity>> {
        if items.is_empty() {
            return Ok(items);
        }
        let model = self.get().await?;
        let documents: Vec<String> = items.iter().map(|s| s.document.content.clone()).collect();
        let query = query.to_string();
        let results =
            tokio::task::spawn_blocking(move || model.rerank(query, documents, false, None))
                .await
                .map_err(|e| RagError::EmbeddingError(format!("join error: {e}")))?
                .map_err(|e| RagError::EmbeddingError(format!("fastembed rerank: {e}")))?;

        // results are sorted by score descending; remap onto the input items.
        let mut reranked = Vec::with_capacity(results.len());
        for r in results {
            if let Some(item) = items.get(r.index) {
                let mut scored = item.clone();
                scored.score = r.score;
                reranked.push(scored);
            }
        }
        Ok(reranked)
    }
}

/// CLIP-style local image embeddings (requires `image-embeddings` feature).
#[cfg(feature = "image-embeddings")]
pub struct FastEmbedImageEmbeddingModel {
    model: Mutex<Option<Arc<fastembed::ImageEmbedding>>>,
    options: fastembed::ImageInitOptions,
}

#[cfg(feature = "image-embeddings")]
impl FastEmbedImageEmbeddingModel {
    /// Default image model (CLIP ViT-B/32 vision).
    pub fn new() -> Self {
        Self::with_model(fastembed::ImageEmbeddingModel::ClipVitB32)
    }

    /// Create an image embedding model with a specific model.
    pub fn with_model(model: fastembed::ImageEmbeddingModel) -> Self {
        Self::with_options(fastembed::ImageInitOptions::new(model))
    }

    /// Create an image embedding model from full `ImageInitOptions`.
    pub fn with_options(options: fastembed::ImageInitOptions) -> Self {
        Self {
            model: Mutex::new(None),
            options,
        }
    }

    async fn get(&self) -> Result<Arc<fastembed::ImageEmbedding>> {
        let mut guard = self.model.lock().await;
        if let Some(m) = guard.as_ref() {
            return Ok(Arc::clone(m));
        }
        let m = Arc::new(
            fastembed::ImageEmbedding::try_new(self.options.clone())
                .map_err(|e| RagError::EmbeddingError(format!("fastembed image init: {e}")))?,
        );
        *guard = Some(Arc::clone(&m));
        Ok(m)
    }

    /// Embed a single image from raw bytes (PNG/JPEG).
    pub async fn embed_image(&self, image_bytes: Vec<u8>) -> Result<Vec<f32>> {
        let model = self.get().await?;
        let embedding = tokio::task::spawn_blocking(move || {
            let images: &[&[u8]] = &[&image_bytes];
            model.embed_bytes(images, None)
        })
        .await
        .map_err(|e| RagError::EmbeddingError(format!("join error: {e}")))?
        .map_err(|e| RagError::EmbeddingError(format!("fastembed image embed: {e}")))?;
        embedding
            .into_iter()
            .next()
            .ok_or_else(|| RagError::EmbeddingError("no image embedding returned".to_string()))
    }
}

#[cfg(feature = "image-embeddings")]
impl Default for FastEmbedImageEmbeddingModel {
    fn default() -> Self {
        Self::new()
    }
}
