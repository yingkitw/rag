//! Offline demo of `GraphStore`: nodes, edges, lookup by name, BFS, and stats.
//!
//! Run: `cargo run --example graph_store_basic`

use rag::graph::{GraphEdge, GraphNode, GraphStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GraphStore basics\n");

    let g = GraphStore::new();

    let alice = GraphNode::new("Alice".to_string(), "person".to_string());
    let alice_id = alice.id.clone();
    let bob = GraphNode::new("Bob".to_string(), "person".to_string());
    let bob_id = bob.id.clone();
    let rag = GraphNode::new("RAG".to_string(), "topic".to_string());
    let rag_id = rag.id.clone();

    g.add_node(alice)?;
    g.add_node(bob)?;
    g.add_node(rag)?;

    g.add_edge(GraphEdge::new(
        alice_id.clone(),
        bob_id.clone(),
        "collaborates_with".to_string(),
    ))?;
    g.add_edge(GraphEdge::new(
        alice_id.clone(),
        rag_id.clone(),
        "works_on".to_string(),
    ))?;
    g.add_edge(GraphEdge::new(
        bob_id.clone(),
        rag_id.clone(),
        "works_on".to_string(),
    ))?;

    let alice_lookup = g.get_node_by_name("Alice").expect("Alice by name");
    println!("Lookup by name: {} ({})", alice_lookup.name, alice_lookup.label);

    let nbrs = g.neighbors(&alice_id);
    println!("Alice degree: {}", g.degree(&alice_id));
    println!("Alice neighbor names: {:?}", nbrs.iter().map(|n| &n.name).collect::<Vec<_>>());

    let within_one = g.bfs(&alice_id, 1);
    println!(
        "BFS depth<=1 from Alice: {:?}",
        within_one.iter().map(|n| &n.name).collect::<Vec<_>>()
    );

    println!(
        "\nGraph stats: {} nodes, {} edges, density {:.4}",
        g.node_count(),
        g.edge_count(),
        g.density()
    );

    let communities = g.detect_communities();
    println!("Communities detected: {}", communities.len());

    Ok(())
}
