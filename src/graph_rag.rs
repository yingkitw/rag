use crate::chunker::TextChunker;
use crate::embeddings::EmbeddingModel;
use crate::errors::Result;
use crate::graph::{GraphEdge, GraphNode, GraphStore};
use crate::vector_store::{Document, VectorStore};
use dashmap::DashMap;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub label: String,
}

pub trait EntityExtractor: Send + Sync {
    fn extract_entities(&self, text: &str) -> Vec<ExtractedEntity>;
}

pub struct SimpleEntityExtractor {
    min_word_length: usize,
}

impl Default for SimpleEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleEntityExtractor {
    pub fn new() -> Self {
        Self {
            min_word_length: 2,
        }
    }
}

impl EntityExtractor for SimpleEntityExtractor {
    fn extract_entities(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut seen = HashSet::new();
        let mut entities = Vec::new();

        for entity in extract_quoted_strings(text) {
            let key = entity.to_lowercase();
            if seen.insert(key) {
                entities.push(ExtractedEntity {
                    name: entity,
                    label: "quoted_term".to_string(),
                });
            }
        }

        for entity in extract_acronyms(text) {
            let key = entity.to_lowercase();
            if seen.insert(key) {
                entities.push(ExtractedEntity {
                    name: entity,
                    label: "acronym".to_string(),
                });
            }
        }

        for entity in extract_proper_nouns(text) {
            let key = entity.to_lowercase();
            if seen.insert(key) && entity.len() >= self.min_word_length {
                entities.push(ExtractedEntity {
                    name: entity,
                    label: "proper_noun".to_string(),
                });
            }
        }

        entities
    }
}

fn extract_quoted_strings(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let quote_char = match chars[i] {
            '"' | '\'' | '`' => chars[i],
            _ => {
                i += 1;
                continue;
            }
        };

        let start = i + 1;
        let mut end = start;
        while end < chars.len() && chars[end] != quote_char {
            end += 1;
        }

        if end < chars.len() && end > start {
            let s: String = chars[start..end].iter().collect();
            let trimmed = s.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() >= 2 {
                results.push(trimmed);
            }
        }

        i = end + 1;
    }

    results
}

fn extract_acronyms(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .collect();
        if cleaned.len() >= 2 && cleaned.len() <= 8 && cleaned.chars().all(|c| c.is_ascii_uppercase())
        {
            results.push(cleaned);
        }
    }
    results
}

fn extract_proper_nouns(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?' || c == '\n').collect();

    let sentence_starters = [
        "the", "a", "an", "this", "that", "it", "there", "here", "when", "where",
        "how", "why", "what", "which", "who", "if", "but", "and", "or", "so",
        "yet", "for", "as", "in", "on", "at", "by", "to", "from", "with",
        "these", "those", "its", "his", "her", "my", "your", "our", "their",
        "all", "some", "any", "each", "every", "no", "not",
    ];

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        let mut current_seq: Vec<String> = Vec::new();

        for (idx, word) in words.iter().enumerate() {
            let is_capitalized = word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);

            let lower = word.to_lowercase();
            let lower_clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
            let is_stop = lower_clean.is_empty()
                || sentence_starters.contains(&lower_clean.as_str());

            let is_sentence_start = idx == 0;

            if is_capitalized && !is_stop && !is_sentence_start {
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
                    .collect();
                if !cleaned.is_empty() {
                    current_seq.push(cleaned);
                }
            } else if is_sentence_start && is_capitalized && !is_stop {
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
                    .collect();
                if !cleaned.is_empty() {
                    current_seq.push(cleaned);
                }
            } else {
                if current_seq.len() >= 1 {
                    results.push(current_seq.join(" "));
                }
                current_seq.clear();
            }
        }

        if current_seq.len() >= 1 {
            results.push(current_seq.join(" "));
        }
    }

    results
}

pub struct GraphRagEngine<E, T, V>
where
    E: EntityExtractor,
    T: EmbeddingModel,
    V: VectorStore,
{
    entity_extractor: E,
    embedding_model: T,
    vector_store: V,
    graph: GraphStore,
    chunker: Box<dyn TextChunker>,
    top_k: usize,
    graph_depth: usize,
    entity_chunks: DashMap<String, HashSet<String>>,
    chunk_entities: DashMap<String, HashSet<String>>,
}

impl<E, T, V> GraphRagEngine<E, T, V>
where
    E: EntityExtractor,
    T: EmbeddingModel,
    V: VectorStore,
{
    pub fn new(entity_extractor: E, embedding_model: T, vector_store: V) -> Self {
        Self {
            entity_extractor,
            embedding_model,
            vector_store,
            graph: GraphStore::new(),
            chunker: Box::new(crate::chunker::ParagraphChunker),
            top_k: 5,
            graph_depth: 2,
            entity_chunks: DashMap::new(),
            chunk_entities: DashMap::new(),
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

    pub fn with_graph_depth(mut self, depth: usize) -> Self {
        self.graph_depth = depth;
        self
    }

    pub async fn add_document(&self, content: String) -> Result<Vec<String>> {
        let chunks = self.chunker.chunk(&content)?;
        let chunk_embeddings = self.embedding_model.embed(chunks.clone()).await?;

        let mut doc_ids = Vec::new();

        for (chunk_text, embedding) in chunks.into_iter().zip(chunk_embeddings.into_iter()) {
            let doc = Document::new(chunk_text.clone()).with_embedding(embedding);
            let doc_id = doc.id.clone();

            let entities = self.entity_extractor.extract_entities(&chunk_text);

            for entity in &entities {
                let node = match self.graph.get_node_by_name(&entity.name) {
                    Some(existing) => existing,
                    None => {
                        let node =
                            GraphNode::new(entity.name.clone(), entity.label.clone());
                        let id = node.id.clone();
                        self.graph.add_node(node)?;
                        self.graph.get_node(&id).unwrap()
                    }
                };

                self.entity_chunks
                    .entry(node.name.clone())
                    .or_insert_with(HashSet::new)
                    .insert(doc_id.clone());

                self.chunk_entities
                    .entry(doc_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(node.name.clone());
            }

            let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
            for i in 0..entity_names.len() {
                for j in (i + 1)..entity_names.len() {
                    if let (Some(src), Some(tgt)) = (
                        self.graph.get_node_by_name(&entity_names[i]),
                        self.graph.get_node_by_name(&entity_names[j]),
                    ) {
                        let edge = GraphEdge::new(
                            src.id.clone(),
                            tgt.id.clone(),
                            "co_occurs".to_string(),
                        )
                        .with_weight(1.0);

                        if let Some(existing) = self
                            .graph
                            .find_edge(&src.id, &tgt.id, "co_occurs")
                        {
                            let new_weight = existing.weight + 1.0;
                            let new_edge = GraphEdge::new(
                                src.id.clone(),
                                tgt.id.clone(),
                                "co_occurs".to_string(),
                            )
                            .with_weight(new_weight);
                            self.graph.upsert_edge(new_edge)?;
                        } else {
                            self.graph.add_edge(edge)?;
                        }

                        let edge_rev = GraphEdge::new(
                            tgt.id.clone(),
                            src.id.clone(),
                            "co_occurs".to_string(),
                        )
                        .with_weight(1.0);

                        if self
                            .graph
                            .find_edge(&tgt.id, &src.id, "co_occurs")
                            .is_none()
                        {
                            if let Some(existing) = self
                                .graph
                                .find_edge(&tgt.id, &src.id, "co_occurs")
                            {
                                let new_weight = existing.weight + 1.0;
                                let new_edge = GraphEdge::new(
                                    tgt.id.clone(),
                                    src.id.clone(),
                                    "co_occurs".to_string(),
                                )
                                .with_weight(new_weight);
                                self.graph.upsert_edge(new_edge)?;
                            } else {
                                let _ = self.graph.add_edge(edge_rev);
                            }
                        }
                    }
                }
            }

            self.vector_store.add(doc).await?;
            doc_ids.push(doc_id);
        }

        Ok(doc_ids)
    }

    pub async fn query(&self, query: &str) -> Result<Vec<GraphRagResult>> {
        let query_embedding = self.embedding_model.embed_single(query).await?;
        let vector_results = self
            .vector_store
            .search(&query_embedding, self.top_k)
            .await?;

        let query_entities = self.entity_extractor.extract_entities(query);
        let mut graph_chunk_ids = HashSet::new();

        for entity in &query_entities {
            if let Some(node) = self.graph.get_node_by_name(&entity.name) {
                let reachable = self.graph.bfs(&node.id, self.graph_depth);
                for neighbor in reachable {
                    if let Some(chunks) = self.entity_chunks.get(&neighbor.name) {
                        for chunk_id in chunks.value().iter() {
                            graph_chunk_ids.insert(chunk_id.clone());
                        }
                    }
                }

                if let Some(chunks) = self.entity_chunks.get(&node.name) {
                    for chunk_id in chunks.value().iter() {
                        graph_chunk_ids.insert(chunk_id.clone());
                    }
                }
            }
        }

        let mut seen_ids = HashSet::new();
        let mut results = Vec::new();

        for sim in &vector_results {
            seen_ids.insert(sim.document.id.clone());
            let entities = self
                .chunk_entities
                .get(&sim.document.id)
                .map(|e| e.value().iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();

            results.push(GraphRagResult {
                content: sim.document.content.clone(),
                score: sim.score,
                source: "vector".to_string(),
                entities,
            });
        }

        for chunk_id in graph_chunk_ids {
            if seen_ids.insert(chunk_id.clone()) {
                if let Some(doc) = self.vector_store.get(&chunk_id).await? {
                    let entities = self
                        .chunk_entities
                        .get(&chunk_id)
                        .map(|e| e.value().iter().cloned().collect::<Vec<_>>())
                        .unwrap_or_default();

                    results.push(GraphRagResult {
                        content: doc.content,
                        score: 0.0,
                        source: "graph".to_string(),
                        entities,
                    });
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);

        Ok(results)
    }

    pub fn graph_store(&self) -> &GraphStore {
        &self.graph
    }

    pub fn vector_store(&self) -> &V {
        &self.vector_store
    }

    pub fn get_entity_info(&self, name: &str) -> Option<EntityInfo> {
        let node = self.graph.get_node_by_name(name)?;
        let neighbors = self.graph.neighbors(&node.id);
        let neighbor_names: Vec<String> = neighbors.iter().map(|n| n.name.clone()).collect();
        let chunks = self
            .entity_chunks
            .get(&*name)
            .map(|e| e.value().len())
            .unwrap_or(0);

        Some(EntityInfo {
            name: node.name,
            label: node.label,
            degree: self.graph.degree(&node.id),
            neighbors: neighbor_names,
            chunk_count: chunks,
        })
    }

    pub fn get_communities(&self) -> Vec<crate::graph::Community> {
        self.graph.detect_communities()
    }

    pub fn graph_info(&self) -> GraphInfo {
        let communities = self.graph.detect_communities();
        GraphInfo {
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            density: self.graph.density(),
            community_count: communities.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphRagResult {
    pub content: String,
    pub score: f32,
    pub source: String,
    pub entities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub name: String,
    pub label: String,
    pub degree: usize,
    pub neighbors: Vec<String>,
    pub chunk_count: usize,
}

#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub community_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_quoted_strings() {
        let text = r#"The concept of "GraphRAG" is related to 'knowledge graph' and `vector database`."#;
        let results = extract_quoted_strings(text);
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"GraphRAG".to_string()));
        assert!(results.contains(&"knowledge graph".to_string()));
        assert!(results.contains(&"vector database".to_string()));
    }

    #[test]
    fn test_extract_acronyms() {
        let text = "RAG and LLM are used in NLP systems.";
        let results = extract_acronyms(text);
        assert!(results.contains(&"RAG".to_string()));
        assert!(results.contains(&"LLM".to_string()));
        assert!(results.contains(&"NLP".to_string()));
    }

    #[test]
    fn test_extract_proper_nouns() {
        let text = "Alice went to New York. Bob lives in San Francisco.";
        let results = extract_proper_nouns(text);
        assert!(results.iter().any(|e| e.contains("Alice")));
        assert!(results.iter().any(|e| e.contains("New York")));
        assert!(results.iter().any(|e| e.contains("Bob")));
        assert!(results.iter().any(|e| e.contains("San Francisco")));
    }

    #[test]
    fn test_simple_entity_extractor() {
        let extractor = SimpleEntityExtractor::new();
        let text = r#"OpenAI released "GPT-4" which uses the RAG technique. Microsoft GraphRAG combines knowledge graphs with LLM technology."#;
        let entities = extractor.extract_entities(text);

        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.iter().any(|n| *n == "GPT-4"));
        assert!(names.iter().any(|n| *n == "RAG"));
        assert!(names.iter().any(|n| *n == "LLM"));
    }

    #[test]
    fn test_extract_empty_text() {
        let extractor = SimpleEntityExtractor::new();
        let entities = extractor.extract_entities("");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_no_entities() {
        let extractor = SimpleEntityExtractor::new();
        let entities = extractor.extract_entities("the quick brown fox jumps over the lazy dog");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_deduplication() {
        let extractor = SimpleEntityExtractor::new();
        let text = "RAG is great. RAG is powerful.";
        let entities = extractor.extract_entities(text);
        let rag_count = entities.iter().filter(|e| e.name == "RAG").count();
        assert_eq!(rag_count, 1);
    }
}
