//! Integration tests for the new capabilities:
//! evaluation metrics, recursive/semantic/structural chunking, Int8
//! quantization, SIMD distances, and the write-ahead log.

use rag::{
    chunker::{RecursiveChunker, SemanticChunker, StructuralChunker, TextChunker},
    eval::{
        EvalReport, average_precision, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
        relevance_set,
    },
    quantize::{QuantizationParams, QuantizedIndex},
    simd,
    vector_store::Document,
    wal::{WalOp, WriteAheadLog, apply_ops},
};
use std::collections::HashSet;
use tempfile::TempDir;

// ---------------- Evaluation metrics ----------------

#[test]
fn eval_metrics_end_to_end() {
    let relevant: HashSet<String> =
        relevance_set(["d1".to_string(), "d3".to_string(), "d7".to_string()]);
    let retrieved = vec![
        "d1".to_string(),
        "d2".to_string(),
        "d7".to_string(),
        "d4".to_string(),
    ];

    assert!((recall_at_k(&retrieved, &relevant, 4) - (2.0 / 3.0)).abs() < 1e-9);
    assert!((precision_at_k(&retrieved, &relevant, 4) - 0.5).abs() < 1e-9);
    assert_eq!(reciprocal_rank(&retrieved, &relevant), 1.0);
    assert!(average_precision(&retrieved, &relevant) > 0.5);

    let grades = vec![3.0, 0.0, 2.0, 0.0];
    let ideal = vec![3.0, 2.0];
    let ndcg = ndcg_at_k(&grades, &ideal, 4);
    assert!(ndcg > 0.0 && ndcg <= 1.0 + 1e-9);
}

#[tokio::test]
async fn eval_evaluate_aggregates_multiple_queries() {
    let queries = vec![
        (
            "rust memory".to_string(),
            vec!["a".to_string(), "b".to_string()],
        ),
        ("paris tower".to_string(), vec!["c".to_string()]),
    ];
    let report = rag::eval::evaluate(queries, 2, |q| async move {
        if q.contains("rust") {
            vec!["a".to_string(), "x".to_string()]
        } else {
            vec!["c".to_string(), "y".to_string()]
        }
    })
    .await;
    assert_eq!(report.n_queries, 2);
    assert!(report.recall > 0.0 && report.recall <= 1.0);
    assert!((report.mrr - 1.0).abs() < 1e-9);
    let empty = EvalReport::default();
    assert_eq!(empty.n_queries, 0);
}

// ---------------- Chunkers ----------------

#[test]
fn recursive_chunker_splits_long_text() {
    let chunker = RecursiveChunker::new(50, 0);
    let text = "alpha beta gamma delta.\n\nepsilon zeta eta theta.\n\niota kappa lambda.";
    let chunks = chunker.chunk(text).unwrap();
    assert!(chunks.len() >= 2);
    for c in &chunks {
        assert!(c.len() <= 55);
    }
}

#[test]
fn recursive_chunker_rejects_bad_overlap() {
    let chunker = RecursiveChunker::new(10, 20);
    assert!(chunker.chunk("some text here").is_err());
}

#[test]
fn semantic_chunker_groups_by_topic() {
    let chunker = SemanticChunker::new(500, 0.1);
    let text = "cats meow. cats purr. cats nap.\n\nquantum physics studies particles. particles make waves.";
    let chunks = chunker.chunk(text).unwrap();
    assert!(chunks.len() >= 2);
}

#[test]
fn structural_chunker_keeps_markdown_sections() {
    let chunker = StructuralChunker::new(1000, 0);
    let text = "# Title\nintro.\n## A\ncontent a.\n## B\ncontent b.";
    let chunks = chunker.chunk(text).unwrap();
    assert!(
        chunks
            .iter()
            .any(|c| c.contains("Section A") || c.contains("content a"))
    );
    assert!(chunks.iter().any(|c| c.contains("content b")));
}

#[test]
fn structural_chunker_code_fence_together() {
    let chunker = StructuralChunker::new(5000, 0);
    let text = "# Intro\nprose.\n```rust\nfn main() {}\n```\n## After\nmore prose.";
    let chunks = chunker.chunk(text).unwrap();
    assert!(chunks.iter().any(|c| c.contains("fn main()")));
}

// ---------------- Quantization ----------------

#[test]
fn quantization_roundtrip_preserves_order() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![0.1, 0.1, 0.1],
    ];
    let params = QuantizationParams::fit(&vectors).unwrap();
    let mut index = QuantizedIndex::new(params);
    index.add("x", &[1.0, 0.0, 0.0]);
    index.add("y", &[0.0, 1.0, 0.0]);
    index.add("z", &[1.0, 1.0, 0.0]);

    let res = index.search(&[1.0, 0.0, 0.0], 3);
    assert_eq!(res.len(), 3);
    // "y" is orthogonal to the query and must rank last; the aligned
    // vectors "x"/"z" tie after dequantization, so only require top-2 set.
    assert_eq!(res.last().unwrap().0, "y");
    let top2: Vec<&str> = res.iter().take(2).map(|(id, _)| id.as_str()).collect();
    assert!(top2.contains(&"x"));
}

#[test]
fn quantization_errors_on_empty() {
    assert!(QuantizationParams::fit(&Vec::<Vec<f32>>::new()).is_err());
}

// ---------------- SIMD distances ----------------

#[test]
fn simd_matches_scalar_results() {
    let a: Vec<f32> = (0..300).map(|i| (i as f32) * 0.1 - 15.0).collect();
    let b: Vec<f32> = (0..300).map(|i| (i as f32) * 0.05).collect();

    let scalar_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert!((simd::dot_product(&a, &b) - scalar_dot).abs() < 1e-2);

    let scalar_euc: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    assert!((simd::euclidean_distance(&a, &b) - scalar_euc).abs() < 1e-2);

    assert!(simd::cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
}

// ---------------- Write-ahead log ----------------

#[test]
fn wal_append_replay_reconstructs_state() {
    let dir = TempDir::new().unwrap();
    let wal = WriteAheadLog::new(dir.path().join("wal.jsonl"));
    wal.append(&WalOp::Put(Document::with_id(
        "a".to_string(),
        "one".to_string(),
    )))
    .unwrap();
    wal.append_batch(&[
        WalOp::Put(Document::with_id("b".to_string(), "two".to_string())),
        WalOp::Delete("a".to_string()),
    ])
    .unwrap();

    let ops = wal.replay().unwrap();
    let state = apply_ops(&ops);
    assert!(!state.contains_key("a"));
    assert_eq!(state.get("b").unwrap().content, "two");
}

#[test]
fn wal_truncate_after_checkpoint() {
    let dir = TempDir::new().unwrap();
    let wal = WriteAheadLog::new(dir.path().join("wal.jsonl"));
    wal.append(&WalOp::Put(Document::with_id(
        "a".to_string(),
        "one".to_string(),
    )))
    .unwrap();
    assert!(wal.size() > 0);
    wal.truncate().unwrap();
    assert_eq!(wal.size(), 0);
    assert!(wal.replay().unwrap().is_empty());
}

#[test]
fn wal_missing_file_replays_empty() {
    let wal = WriteAheadLog::new("/nonexistent/path/wal.jsonl");
    assert!(wal.replay().unwrap().is_empty());
}

// ---------------- Compressed persistence (feature-gated) ----------------

#[cfg(feature = "compress")]
#[test]
fn compress_roundtrip_via_lib() {
    use rag::compress::{compress_bytes, decompress_bytes};
    let data = b"repeating repeating repeating".repeat(500);
    let compressed = compress_bytes(&data, 19).unwrap();
    assert!(compressed.len() < data.len());
    let back = decompress_bytes(&compressed).unwrap();
    assert_eq!(back, data);
}
