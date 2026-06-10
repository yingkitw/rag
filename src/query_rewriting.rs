//! LLM-based query rewriting and multi-query generation.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::errors::{RagError, Result};

#[derive(Clone)]
pub struct QueryRewriter {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl QueryRewriter {
    pub fn openai(api_key: String) -> Self {
        Self { client: Client::new(), api_key, model: "gpt-4o-mini".to_string(), base_url: "https://api.openai.com/v1".to_string() }
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

impl QueryRewriter {
    /// Generate `n` rewritten variants of the user query.
    pub async fn rewrite(&self, query: &str, n: usize) -> Result<Vec<String>> {
        let prompt = format!(
            "Generate {} alternative search queries that express the same intent as: \"{}\". Output one per line, no numbering.",
            n, query
        );
        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![Message { role: "user".to_string(), content: prompt }],
        };
        let resp = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send().await?;
        if !resp.status().is_success() {
            return Err(RagError::EmbeddingError(resp.text().await?));
        }
        let data: ChatResponse = resp.json().await?;
        let text = data.choices.into_iter().next().map(|c| c.message.content).unwrap_or_default();
        let variants: Vec<String> = text.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).take(n).collect();
        Ok(variants)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewriter_struct_exists() {
        // Basic compilation check
        let _ = QueryRewriter::openai("test".to_string());
    }
}
