#[cfg(feature = "http")]
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::future::Future;

use crate::errors::{RagError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub text: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
    pub model: String,
}

pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, texts: Vec<String>) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;
    fn embed_single(&self, text: &str) -> impl Future<Output = Result<Vec<f32>>> + Send {
        async move {
            let embeddings = self.embed(vec![text.to_string()]).await?;
            Ok(embeddings.into_iter().next().ok_or(RagError::EmbeddingError(
                "No embedding returned".to_string(),
            ))?)
        }
    }
}

#[cfg(feature = "http")]
#[derive(Clone)]
pub struct OpenAIEmbeddingModel {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

#[cfg(feature = "http")]
impl OpenAIEmbeddingModel {
    pub fn new(api_key: String) -> Self {
        Self::with_model(api_key, "text-embedding-ada-002".to_string())
    }

    pub fn with_model(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[cfg(feature = "http")]
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    input: Vec<String>,
    model: String,
}

#[cfg(feature = "http")]
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    data: Vec<OpenAIEmbeddingData>,
    #[allow(dead_code)]
    model: String,
}

#[cfg(feature = "http")]
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(feature = "http")]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIRequest {
            input: texts.clone(),
            model: self.model.clone(),
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RagError::EmbeddingError(error_text));
        }

        let openai_response: OpenAIResponse = response.json().await?;

        Ok(openai_response.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(feature = "http")]
#[derive(Clone)]
pub struct OllamaEmbeddingModel {
    client: Client,
    model: String,
    base_url: String,
}

#[cfg(feature = "http")]
impl OllamaEmbeddingModel {
    pub fn new(model: String) -> Self {
        Self {
            client: Client::new(),
            model,
            base_url: "http://localhost:11434".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[cfg(feature = "http")]
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
}

#[cfg(feature = "http")]
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    embedding: Vec<f32>,
}

#[cfg(feature = "http")]
impl EmbeddingModel for OllamaEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let request = OllamaRequest {
                model: self.model.clone(),
                prompt: text,
            };

            let response = self
                .client
                .post(format!("{}/api/embeddings", self.base_url))
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(RagError::EmbeddingError(error_text));
            }

            let ollama_response: OllamaResponse = response.json().await?;
            embeddings.push(ollama_response.embedding);
        }

        Ok(embeddings)
    }
}

/// HTTP embeddings against an OpenAI-compatible `/embeddings` JSON API.
///
/// Sends `POST {base_url}{embeddings_path}` with body `{ "input": string[], "model": string }`.
#[cfg(feature = "http")]
#[derive(Clone)]
pub struct HttpEmbeddingModel {
    client: Client,
    api_key: Option<String>,
    base_url: String,
    model: String,
    embeddings_path: String,
}

#[cfg(feature = "http")]
impl HttpEmbeddingModel {
    pub fn openai_compatible(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key),
            base_url: "https://api.openai.com/v1".to_string(),
            model,
            embeddings_path: "/embeddings".to_string(),
        }
    }

    pub fn without_api_key(mut self) -> Self {
        self.api_key = None;
        self
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_embeddings_path(mut self, path: String) -> Self {
        self.embeddings_path = path;
        self
    }
}

#[cfg(feature = "http")]
impl EmbeddingModel for HttpEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIRequest {
            input: texts.clone(),
            model: self.model.clone(),
        };
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            self.embeddings_path.trim_start_matches('/')
        );
        let mut req = self.client.post(url).json(&request);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        let response = req.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RagError::EmbeddingError(error_text));
        }

        let openai_response: OpenAIResponse = response.json().await?;
        Ok(openai_response.data.into_iter().map(|d| d.embedding).collect())
    }
}