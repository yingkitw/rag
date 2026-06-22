//! SQLite-backed vector store implementing [`VectorStore`].
//!
//! Stores documents, metadata, and embeddings in a local SQLite file.
//! Search is brute-force in Rust (no vector index inside SQLite).

use crate::errors::{RagError, Result};
use crate::index::DistanceMetric;
use crate::vector_store::{Document, MetadataFilter, Similarity, VectorStore};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct SqliteVectorStore {
    conn: Arc<Mutex<Connection>>,
    metric: DistanceMetric,
}

impl SqliteVectorStore {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| RagError::IoError(std::io::Error::other(
            format!("Failed to open SQLite: {}", e),
        )))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            metric: DistanceMetric::Cosine,
        };
        store.init_schema()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| RagError::IoError(std::io::Error::other(
            format!("Failed to open in-memory SQLite: {}", e),
        )))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            metric: DistanceMetric::Cosine,
        };
        store.init_schema()?;
        Ok(store)
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT
            )",
            [],
        )
        .map_err(|e| RagError::VectorStoreError(format!("Schema init failed: {}", e)))?;
        Ok(())
    }

    fn row_to_doc(row: &rusqlite::Row) -> rusqlite::Result<Document> {
        let id: String = row.get(0)?;
        let content: String = row.get(1)?;
        let metadata_json: String = row.get(2)?;
        let embedding_json: Option<String> = row.get(3)?;

        let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();
        let embedding = embedding_json.and_then(|s| serde_json::from_str(&s).ok());

        Ok(Document {
            id,
            content,
            metadata,
            embedding,
        })
    }
}

#[allow(async_fn_in_trait)]
impl VectorStore for SqliteVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let metadata_json = serde_json::to_string(&document.metadata)
            .map_err(RagError::JsonError)?;
        let embedding_json = document
            .embedding
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(RagError::JsonError)?;

        conn.execute(
            "INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
             VALUES (?1, ?2, ?3, ?4)",
            params![document.id, document.content, metadata_json, embedding_json],
        )
        .map_err(|e| RagError::VectorStoreError(format!("Insert failed: {}", e)))?;
        Ok(())
    }

    async fn add_batch(&self, documents: Vec<Document>) -> Result<()> {
        for doc in documents {
            self.add(doc).await?;
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Similarity>> {
        self.search_with_filter(query, top_k, &MetadataFilter::new()).await
    }

    async fn search_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<Similarity>> {
        let all = self.list(usize::MAX, 0).await?;
        let mut scored: Vec<Similarity> = all
            .into_iter()
            .filter(|doc| filter.matches(&doc.metadata))
            .filter_map(|doc| {
                doc.embedding.clone().map(|emb| {
                    let score = self.metric.similarity(query, &emb);
                    Similarity { document: doc, score }
                })
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.search(q, top_k).await?);
        }
        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        let conn = self.conn.lock().unwrap();
        let doc = conn
            .query_row(
                "SELECT id, content, metadata, embedding FROM documents WHERE id = ?1",
                params![id],
                Self::row_to_doc,
            )
            .optional()
            .map_err(|e| RagError::VectorStoreError(format!("Get failed: {}", e)))?;
        Ok(doc)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let changed = conn
            .execute("DELETE FROM documents WHERE id = ?1", params![id])
            .map_err(|e| RagError::VectorStoreError(format!("Delete failed: {}", e)))?;
        Ok(changed > 0)
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        let mut total = 0usize;
        for id in ids {
            if self.delete(&id).await? {
                total += 1;
            }
        }
        Ok(total)
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM documents", [])
            .map_err(|e| RagError::VectorStoreError(format!("Clear failed: {}", e)))?;
        Ok(())
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, content, metadata, embedding FROM documents LIMIT ?1 OFFSET ?2")
            .map_err(|e| RagError::VectorStoreError(format!("Prepare failed: {}", e)))?;
        let docs: Vec<Document> = stmt
            .query_map(params![limit as i64, offset as i64], Self::row_to_doc)
            .map_err(|e| RagError::VectorStoreError(format!("Query failed: {}", e)))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| RagError::VectorStoreError(format!("Row mapping failed: {}", e)))?;
        Ok(docs)
    }

    async fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(|e| RagError::VectorStoreError(format!("Count failed: {}", e)))?;
        Ok(count as usize)
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_add_and_get() {
        let store = SqliteVectorStore::open_in_memory().unwrap();
        let doc = Document::new("hello world".to_string())
            .with_embedding(vec![1.0, 0.0, 0.0]);
        let id = doc.id.clone();
        store.add(doc).await.unwrap();

        let fetched = store.get(&id).await.unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().content, "hello world");
    }

    #[tokio::test]
    async fn test_sqlite_search() {
        let store = SqliteVectorStore::open_in_memory().unwrap();
        store.add(Document::new("Rust is fast".to_string()).with_embedding(vec![1.0, 0.0, 0.0])).await.unwrap();
        store.add(Document::new("Python is slow".to_string()).with_embedding(vec![0.0, 1.0, 0.0])).await.unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.content, "Rust is fast");
    }

    #[tokio::test]
    async fn test_sqlite_delete_and_count() {
        let store = SqliteVectorStore::open_in_memory().unwrap();
        let doc = Document::new("test".to_string());
        let id = doc.id.clone();
        store.add(doc).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        assert!(store.delete(&id).await.unwrap());
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(!store.delete(&id).await.unwrap());
    }

    #[tokio::test]
    async fn test_sqlite_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");

        let doc_id = {
            let store = SqliteVectorStore::open(&path).unwrap();
            let doc = Document::new("persistent content".to_string()).with_embedding(vec![0.5, 0.5]);
            let id = doc.id.clone();
            store.add(doc).await.unwrap();
            id
        };

        let store2 = SqliteVectorStore::open(&path).unwrap();
        let fetched = store2.get(&doc_id).await.unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().content, "persistent content");
    }
}
