//! Contextual retrieval: rewrite chunks with surrounding document context before embedding.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::errors::{RagError, Result};

#[derive(Clone)]
pub struct ContextualRetrieval {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl ContextualRetrieval {
    pub fn openai(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

impl ContextualRetrieval {
    /// Rewrite a chunk with document context to make it self-contained for embedding.
    pub async fn rewrite(&self, chunk: &str, document_context: &str) -> Result<String> {
        let prompt = format!(
            "Given the following document context, rewrite this chunk to be self-contained and meaningful for semantic search.\n\nDocument context:\n{}\n\nChunk:\n{}\n\nRewritten chunk:",
            document_context, chunk
        );
        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt,
            }],
        };
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(RagError::EmbeddingError(resp.text().await?));
        }
        let data: ChatResponse = resp.json().await?;
        let text = data
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();
        Ok(text.trim().to_string())
    }

    /// Rewrite a batch of chunks with shared document context.
    pub async fn rewrite_batch(
        &self,
        chunks: &[String],
        document_context: &str,
    ) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            out.push(self.rewrite(chunk, document_context).await?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_struct_exists() {
        let _ = ContextualRetrieval::openai("test".to_string());
    }
}
