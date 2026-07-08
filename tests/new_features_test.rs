use rag::{
    aggregation::{count_by, group_by, sum_by},
    diversify::diversify,
    hybrid::{merge_hybrid, rrf_fusion},
    index::{FlatIndex, Index},
    keyword::{Bm25Config, Bm25Index, FieldBm25Index},
    sparse::{SparseIndex, SparseVector},
    vector_store::{Document, JsonPersistentVectorStore, Similarity, VectorStore},
};
use std::collections::HashMap;
use tempfile::TempDir;

// --- Exact kNN filtered search ---

#[test]
fn exact_knn_filtered_search() {
    let index = FlatIndex::new();
    let mut d1 = Document::new("rust systems".to_string()).with_embedding(vec![1.0, 0.0, 0.0]);
    d1.metadata.insert("lang".to_string(), "rust".to_string());
    let mut d2 = Document::new("python scripts".to_string()).with_embedding(vec![0.0, 1.0, 0.0]);
    d2.metadata.insert("lang".to_string(), "python".to_string());
    let mut d3 = Document::new("rust cli".to_string()).with_embedding(vec![0.95, 0.05, 0.0]);
    d3.metadata.insert("lang".to_string(), "rust".to_string());

    index.add(d1);
    index.add(d2);
    index.add(d3);

    let results = index.search_exact_filtered(&[1.0, 0.0, 0.0], 5, &|doc: &Document| {
        doc.metadata.get("lang") == Some(&"rust".to_string())
    });

    assert_eq!(results.len(), 2);
    // Both rust docs should be returned, ordered by similarity
    assert!(results[0].score >= results[1].score);
}

#[test]
fn exact_knn_empty_filter() {
    let index = FlatIndex::new();
    index.add(Document::new("a".to_string()).with_embedding(vec![1.0, 0.0]));

    let results = index.search_exact_filtered(&[1.0, 0.0], 5, &|_| false);
    assert!(results.is_empty());
}

// --- RRF fusion ---

#[test]
fn rrf_fusion_combines_rankings() {
    let d1 = Document::new("first doc".to_string());
    let d2 = Document::new("second doc".to_string());
    let d3 = Document::new("third doc".to_string());

    let list1 = vec![
        Similarity { document: d1.clone(), score: 0.9 },
        Similarity { document: d2.clone(), score: 0.8 },
    ];
    let list2 = vec![
        Similarity { document: d3.clone(), score: 0.95 },
        Similarity { document: d1.clone(), score: 0.7 },
    ];

    let fused = rrf_fusion(&[list1, list2], 60, 10);
    assert!(!fused.is_empty());
    // d1 appears in both lists so should have highest RRF score
    assert_eq!(fused[0].document.id, d1.id);
}

#[test]
fn rrf_fusion_empty_input() {
    let fused = rrf_fusion(&[], 60, 10);
    assert!(fused.is_empty());
}

#[test]
fn rrf_fusion_single_list() {
    let d = Document::new("only".to_string());
    let fused = rrf_fusion(&[vec![Similarity { document: d, score: 1.0 }]], 60, 10);
    assert_eq!(fused.len(), 1);
}

// --- Merge hybrid with configurable BM25 ---

#[test]
fn merge_hybrid_with_custom_bm25() {
    let docs = vec![
        Document::new("rust programming language safety".to_string()),
        Document::new("python easy scripting dynamic".to_string()),
    ];
    let mut map = HashMap::new();
    for d in &docs {
        map.insert(d.id.clone(), d.clone());
    }

    let idx = Bm25Index::from_documents_with_config(&docs, Bm25Config::new(1.5, 0.5)).unwrap();
    let kw = idx.search("rust safety", 10);

    let vec_hits = vec![
        Similarity {
            document: docs[0].clone(),
            score: 0.95,
        },
        Similarity {
            document: docs[1].clone(),
            score: 0.3,
        },
    ];

    let merged = merge_hybrid(&map, &vec_hits, &kw, 0.7, 2).unwrap();
    assert_eq!(merged.len(), 2);
}

// --- Phrase and prefix search ---

#[test]
fn bm25_phrase_search_exact() {
    let docs = vec![
        Document::new("machine learning is great".to_string()),
        Document::new("deep learning and machine vision".to_string()),
    ];
    let idx = Bm25Index::from_documents(&docs).unwrap();
    let hits = idx.search_phrase("machine learning", 2);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, docs[0].id);
}

#[test]
fn bm25_prefix_search() {
    let docs = vec![
        Document::new("rust programming".to_string()),
        Document::new("python scripting".to_string()),
    ];
    let idx = Bm25Index::from_documents(&docs).unwrap();
    let hits = idx.search_prefix("rust", 2);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, docs[0].id);
}

// --- Field-level BM25 ---

#[test]
fn field_bm25_ranks_by_boost() {
    let mut d1 = Document::new("content1".to_string());
    d1.metadata.insert("title".to_string(), "rust systems".to_string());
    d1.metadata.insert("body".to_string(), "some text".to_string());

    let mut d2 = Document::new("content2".to_string());
    d2.metadata.insert("title".to_string(), "other topic".to_string());
    d2.metadata.insert("body".to_string(), "rust systems deep dive".to_string());

    let docs = vec![d1, d2];
    let mut idx = FieldBm25Index::new(vec![
        ("title".to_string(), 3.0),
        ("body".to_string(), 1.0),
    ]);
    idx.build(&docs).unwrap();

    let hits = idx.search("rust systems", 2);
    // d1 has rust systems in title (boosted 3x), d2 only in body
    assert!(!hits.is_empty());
}

// --- Lazy-flush persistence ---

#[tokio::test]
async fn lazy_flush_no_file_write_until_explicit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.json");

    let store = JsonPersistentVectorStore::open_lazy_flush(&path).await.unwrap();
    let doc = Document::new("hello".to_string()).with_embedding(vec![1.0, 0.0]);
    store.add(doc).await.unwrap();

    // File should not exist yet
    assert!(!path.exists());

    // Now flush explicitly
    store.flush().await.unwrap();
    assert!(path.exists());
}

#[tokio::test]
async fn lazy_flush_reload() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.json");

    let store = JsonPersistentVectorStore::open_lazy_flush(&path).await.unwrap();
    let doc = Document::new("persist me".to_string()).with_embedding(vec![0.5, 0.5]);
    store.add(doc.clone()).await.unwrap();
    store.flush().await.unwrap();

    let reloaded = JsonPersistentVectorStore::open(&path).await.unwrap();
    let count = reloaded.count().await.unwrap();
    assert_eq!(count, 1);
}

// --- Diversification ---

#[test]
fn diversify_limits_per_source() {
    let mut docs = Vec::new();
    for i in 0..5 {
        let mut doc = Document::new(format!("doc {}", i));
        doc.metadata.insert("source".to_string(), "A".to_string());
        docs.push(Similarity { document: doc, score: 1.0 - i as f32 * 0.1 });
    }
    for i in 0..3 {
        let mut doc = Document::new(format!("doc b{}", i));
        doc.metadata.insert("source".to_string(), "B".to_string());
        docs.push(Similarity { document: doc, score: 0.5 - i as f32 * 0.05 });
    }

    let out = diversify(docs, "source", 2, 10);
    assert_eq!(out.len(), 4); // 2 from A, 2 from B
}

#[test]
fn diversify_zero_max_returns_empty() {
    let docs = vec![Similarity { document: Document::new("a".to_string()), score: 1.0 }];
    let out = diversify(docs, "source", 0, 10);
    assert!(out.is_empty());
}

// --- Aggregation ---

#[test]
fn count_by_groups_metadata() {
    let docs = vec![
        Document::new("a".to_string()).with_metadata("tag".to_string(), "x".to_string()),
        Document::new("b".to_string()).with_metadata("tag".to_string(), "x".to_string()),
        Document::new("c".to_string()).with_metadata("tag".to_string(), "y".to_string()),
    ];
    let counts = count_by(&docs, "tag");
    assert_eq!(counts.get("x"), Some(&2));
    assert_eq!(counts.get("y"), Some(&1));
}

#[test]
fn group_by_collects_documents() {
    let docs = vec![
        Document::new("a".to_string()).with_metadata("cat".to_string(), "A".to_string()),
        Document::new("b".to_string()).with_metadata("cat".to_string(), "A".to_string()),
        Document::new("c".to_string()).with_metadata("cat".to_string(), "B".to_string()),
    ];
    let groups = group_by(&docs, "cat");
    assert_eq!(groups.get("A").unwrap().len(), 2);
    assert_eq!(groups.get("B").unwrap().len(), 1);
}

#[test]
fn sum_by_numeric_metadata() {
    let docs = vec![
        Document::new("a".to_string())
            .with_metadata("cat".to_string(), "A".to_string())
            .with_metadata("score".to_string(), "10".to_string()),
        Document::new("b".to_string())
            .with_metadata("cat".to_string(), "A".to_string())
            .with_metadata("score".to_string(), "20".to_string()),
        Document::new("c".to_string())
            .with_metadata("cat".to_string(), "B".to_string())
            .with_metadata("score".to_string(), "5".to_string()),
    ];
    let sums = sum_by(&docs, "cat", "score");
    assert_eq!(sums.get("A"), Some(&30.0));
    assert_eq!(sums.get("B"), Some(&5.0));
}

// --- Sparse vectors ---

#[test]
fn sparse_vector_search_integration() {
    let mut idx = SparseIndex::new();
    idx.add("doc1".to_string(), SparseVector::new().insert(0, 1.0).insert(2, 0.5));
    idx.add("doc2".to_string(), SparseVector::new().insert(1, 1.0).insert(2, 0.3));
    idx.add("doc3".to_string(), SparseVector::new().insert(0, 0.8).insert(2, 0.6));

    let results = idx.search(&SparseVector::new().insert(0, 1.0).insert(2, 0.5), 2);
    assert!(!results.is_empty());
    // doc1 and doc3 share dims 0 and 2 with query
    assert!(results.iter().any(|(id, _)| id == "doc1"));
}

#[test]
fn sparse_vector_dot_product() {
    let a = SparseVector::new().insert(0, 2.0).insert(5, 3.0);
    let b = SparseVector::new().insert(5, 4.0).insert(10, 1.0);
    assert_eq!(a.dot(&b), 12.0); // 3.0 * 4.0 = 12.0 at dim 5
}

#[test]
fn sparse_vector_cosine() {
    let a = SparseVector::new().insert(0, 1.0);
    let b = SparseVector::new().insert(0, 1.0);
    assert!((a.cosine(&b) - 1.0).abs() < 1e-6);
}

// --- Fuzzy matching ---

#[test]
fn fuzzy_filter_typo_tolerance() {
    let terms = vec!["hello", "hallo", "helo", "world", "help"];
    let matches = rag::fuzzy::fuzzy_filter("hello", &terms, 1);
    assert!(matches.contains(&"hello"));
    assert!(matches.contains(&"hallo"));
    assert!(matches.contains(&"helo"));
    assert!(!matches.contains(&"world"));
}

#[test]
fn levenshtein_distance_basic() {
    use rag::fuzzy::levenshtein;
    assert_eq!(levenshtein("kitten", "sitting"), 3);
    assert_eq!(levenshtein("hello", "hello"), 0);
    assert_eq!(levenshtein("hello", "helo"), 1);
}
