//! Integration-style tests for `GraphRagEngine` without network (deterministic embeddings).

use async_trait::async_trait;
use rag::{
    chunker::ParagraphChunker,
    embeddings::EmbeddingModel,
    graph_rag::{GraphRagEngine, SimpleEntityExtractor},
    vector_store::{InMemoryVectorStore, VectorStore},
};

#[derive(Clone)]
struct DeterministicEmbedder {
    dim: usize,
}

impl DeterministicEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn vector_for(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for (i, byte) in text.as_bytes().iter().enumerate() {
            v[i % self.dim] += *byte as f32;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl EmbeddingModel for DeterministicEmbedder {
    async fn embed(&self, texts: Vec<String>) -> rag::errors::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.vector_for(t)).collect())
    }
}

fn engine() -> GraphRagEngine<SimpleEntityExtractor, DeterministicEmbedder, InMemoryVectorStore> {
    GraphRagEngine::new(
        SimpleEntityExtractor::new(),
        DeterministicEmbedder::new(48),
        InMemoryVectorStore::new(),
    )
    .with_chunker(Box::new(ParagraphChunker))
    .with_top_k(8)
    .with_graph_depth(2)
}

#[tokio::test]
async fn graph_rag_engine_ingest_builds_graph() {
    let e = engine();
    e.add_document("ABRA is a codename. BETA shares the same milestone.".to_string())
        .await
        .unwrap();
    e.add_document("BETA implements the storage layer.".to_string())
        .await
        .unwrap();

    assert!(e.graph_store().node_count() >= 2);
    assert!(e.graph_store().edge_count() >= 1);
    let info = e.graph_info();
    assert!(info.node_count >= 2);
}

#[tokio::test]
async fn graph_rag_query_merges_vector_and_graph_paths() {
    let e = engine();
    // Chunk 1: acronyms ABRA and BETA co-occur.
    e.add_document("ABRA is a codename. BETA shares the same milestone.".to_string())
        .await
        .unwrap();
    // Chunk 2: only BETA — reachable from ABRA via graph through shared entity BETA.
    e.add_document("BETA implements the storage layer for the service.".to_string())
        .await
        .unwrap();
    // Chunk 3: distractor with different wording (still embeds uniquely).
    e.add_document("Gamma curve calibration is unrelated to milestones.".to_string())
        .await
        .unwrap();

    let results = e.query("ABRA milestone codename").await.unwrap();
    let joined = results
        .iter()
        .map(|r| r.content.as_str())
        .collect::<Vec<_>>()
        .join(" | ");

    assert!(
        joined.contains("storage"),
        "expected retrieval to surface the BETA/storage chunk; got: {joined}"
    );
    assert!(results.iter().any(|r| r.source == "vector"));

    let beta = e.get_entity_info("BETA").expect("BETA entity");
    assert!(
        beta.chunk_count >= 2,
        "BETA should appear in multiple chunks; chunk_count={}",
        beta.chunk_count
    );
}

#[tokio::test]
async fn graph_rag_get_entity_info_sees_neighbors() {
    let e = engine();
    e.add_document("Alice and Bob both use RAG at Acme.".to_string())
        .await
        .unwrap();

    let alice = e.get_entity_info("Alice").expect("Alice extracted");
    assert!(!alice.neighbors.is_empty() || alice.degree > 0);
}

#[test]
fn deterministic_embedder_unit_length_for_nontrivial_text() {
    let d = DeterministicEmbedder::new(16);
    let v = d.vector_for("hello world");
    assert_eq!(v.len(), 16);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "expected L2 normalization, norm={norm}"
    );
}

#[tokio::test]
async fn graph_rag_query_empty_store_returns_empty() {
    let e = engine();
    let r = e.query("no documents indexed").await.unwrap();
    assert!(r.is_empty());
}

#[tokio::test]
async fn graph_rag_snapshot_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snap.json");
    let e = engine();
    e.add_document("ABRA test snapshot roundtrip.".to_string())
        .await
        .unwrap();
    e.save_snapshot(&path).await.unwrap();
    let e2 = GraphRagEngine::load_from_snapshot_file(
        &path,
        SimpleEntityExtractor::new(),
        DeterministicEmbedder::new(48),
    )
    .await
    .unwrap();
    assert_eq!(e2.vector_store().count().await.unwrap(), 1);
    let q = e2.query("ABRA").await.unwrap();
    assert!(!q.is_empty());
}
