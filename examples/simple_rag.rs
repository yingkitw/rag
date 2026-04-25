use rag::{
    chunker::FixedSizeChunker,
    embeddings::OllamaEmbeddingModel,
    retriever::Retriever,
    vector_store::{InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simple RAG Example\n");

    let embedding_model = OllamaEmbeddingModel::new("nomic-embed-text".to_string());
    let vector_store = InMemoryVectorStore::new();

    let retriever = Retriever::new(embedding_model, vector_store)
        .with_chunker(Box::new(FixedSizeChunker::new(200, 30)))
        .with_top_k(3);

    let documents = vec![
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by providing them with relevant external context.",
        "The vector database stores embeddings which are numerical representations of text that capture semantic meaning.",
        "Cosine similarity is commonly used to measure the similarity between two vectors in RAG systems.",
    ];

    println!("Adding documents to the vector store...");
    for (i, doc) in documents.iter().enumerate() {
        let ids = retriever.add_document_with_metadata(
            doc.to_string(),
            vec![("section".to_string(), format!("section_{}", i + 1))],
        ).await?;
        println!("Added document {}: {}", i + 1, ids);
    }

    println!("\nTotal documents: {}", retriever.vector_store().count().await?);

    let queries = vec![
        "What is Rust?",
        "How does RAG work?",
        "What is cosine similarity used for?",
    ];

    println!("\nRunning queries:");
    for query in queries {
        println!("\nQuery: {}", query);
        let results = retriever.retrieve_with_scores(query).await?;
        
        for (i, (content, score)) in results.iter().enumerate() {
            println!("  {}. [Score: {:.4}] {}", i + 1, score, content);
        }
    }

    Ok(())
}