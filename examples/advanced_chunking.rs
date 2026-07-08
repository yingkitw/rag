//! Example: Advanced chunking strategies — recursive, semantic, and structural.

use rag::chunker::{RecursiveChunker, SemanticChunker, StructuralChunker, TextChunker};

fn main() {
    let markdown = "# Introduction\n\
        This is the intro. It has some sentences about RAG.\n\
        RAG combines retrieval with generation.\n\n\
        ## Architecture\n\
        The system has a vector store and a graph store.\n\
        They are connected through an engine.\n\n\
        ```rust\n\
        fn main() {\n\
            println!(\"code stays together\");\n\
        }\n\
        ```\n\n\
        ## Conclusion\n\
        This wraps things up.";

    println!("=== Recursive chunker (target 120 chars) ===");
    for (i, chunk) in RecursiveChunker::new(120, 0).chunk(markdown).unwrap().iter().enumerate() {
        println!("[{i}] ({}) {chunk:?}\n", chunk.len());
    }

    println!("=== Semantic chunker (token-overlap grouping) ===");
    for (i, chunk) in SemanticChunker::default().chunk(markdown).unwrap().iter().enumerate() {
        println!("[{i}] {chunk}\n");
    }

    println!("=== Structural chunker (markdown + code aware) ===");
    for (i, chunk) in StructuralChunker::default().chunk(markdown).unwrap().iter().enumerate() {
        println!("[{i}] {chunk}\n");
    }
}
