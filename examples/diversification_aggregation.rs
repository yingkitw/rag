//! Example: Result diversification and metadata aggregation.
//!
//! Diversification prevents too many results from the same source.
//! Aggregation provides group-level statistics over document metadata.

use rag::{
    aggregation::{count_by, group_by, sum_by},
    diversify::diversify,
    vector_store::{Document, Similarity},
};

fn main() {
    // Simulate search results with varied sources
    let mut results = Vec::new();
    for i in 0..5 {
        let mut doc = Document::new(format!("Article about topic {}", i));
        doc.metadata
            .insert("source".to_string(), "blog-a".to_string());
        doc.metadata
            .insert("category".to_string(), "tech".to_string());
        results.push(Similarity {
            document: doc,
            score: 1.0 - i as f32 * 0.05,
        });
    }
    for i in 0..3 {
        let mut doc = Document::new(format!("Paper on research {}", i));
        doc.metadata
            .insert("source".to_string(), "arxiv".to_string());
        doc.metadata
            .insert("category".to_string(), "research".to_string());
        results.push(Similarity {
            document: doc,
            score: 0.7 - i as f32 * 0.1,
        });
    }
    for i in 0..2 {
        let mut doc = Document::new(format!("News item {}", i));
        doc.metadata
            .insert("source".to_string(), "news-site".to_string());
        doc.metadata
            .insert("category".to_string(), "news".to_string());
        results.push(Similarity {
            document: doc,
            score: 0.5 - i as f32 * 0.1,
        });
    }

    println!("Original results: {} items", results.len());
    for r in &results {
        let src = r.document.metadata.get("source").unwrap();
        println!("  {:.2}  [{}]  {}", r.score, src, r.document.content);
    }

    // Diversify: max 2 per source
    let diversified = diversify(results, "source", 2, 10);
    println!(
        "\nDiversified (max 2 per source): {} items",
        diversified.len()
    );
    for r in &diversified {
        let src = r.document.metadata.get("source").unwrap();
        println!("  {:.2}  [{}]  {}", r.score, src, r.document.content);
    }

    // Build fresh docs for aggregation demo
    let docs: Vec<Document> = (0..10)
        .map(|i| {
            let source = if i < 4 {
                "blog-a"
            } else if i < 7 {
                "arxiv"
            } else {
                "news-site"
            };
            let cat = if i < 5 { "tech" } else { "research" };
            let score = (i * 10).to_string();
            Document::new(format!("doc{}", i))
                .with_metadata("source".to_string(), source.to_string())
                .with_metadata("category".to_string(), cat.to_string())
                .with_metadata("score".to_string(), score)
        })
        .collect();

    println!("\n--- Aggregation ---");

    let source_counts = count_by(&docs, "source");
    println!("Count by source:");
    for (source, count) in &source_counts {
        println!("  {} -> {}", source, count);
    }

    let cat_groups = group_by(&docs, "category");
    println!("\nGroup by category:");
    for (cat, group_docs) in &cat_groups {
        println!("  {} -> {} docs", cat, group_docs.len());
    }

    let score_sums = sum_by(&docs, "source", "score");
    println!("\nSum of 'score' by source:");
    for (source, total) in &score_sums {
        println!("  {} -> {}", source, total);
    }
}
