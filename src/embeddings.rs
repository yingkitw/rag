use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

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

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(vec![text.to_string()]).await?;
        Ok(embeddings.into_iter().next().ok_or(RagError::EmbeddingError(
            "No embedding returned".to_string(),
        ))?)
    }
}

#[derive(Clone)]
pub struct OpenAIEmbeddingModel {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

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

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIRequest {
            input: texts.clone(),
            model: self.model.clone(),
        };

        let response = self
            .client
            .post(&format!("{}/embeddings", self.base_url))
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

#[derive(Clone)]
pub struct OllamaEmbeddingModel {
    client: Client,
    model: String,
    base_url: String,
}

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

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    embedding: Vec<f32>,
}

#[async_trait]
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
                .post(&format!("{}/api/embeddings", self.base_url))
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