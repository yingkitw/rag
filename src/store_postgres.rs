//! PostgreSQL-backed vector store implementing [`VectorStore`].
//!
//! Uses the `pgvector` extension: embeddings are stored as `vector` values
//! and similarity search is performed inside Postgres with pgvector
//! distance operators (`<=>`, `<->`, `<#>`, `<+>`).

use crate::errors::{RagError, Result};
use crate::index::DistanceMetric;
use crate::vector_store::{Document, MetadataFilter, Similarity, VectorStore};
use pgvector::Vector;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_postgres::{Client, NoTls};

pub struct PostgresVectorStore {
    client: Arc<Client>,
    table: String,
    metric: DistanceMetric,
}

impl PostgresVectorStore {
    pub async fn connect(conn_str: &str, table: impl Into<String>) -> Result<Self> {
        let (client, connection) = tokio_postgres::connect(conn_str, NoTls)
            .await
            .map_err(|e| {
                RagError::IoError(std::io::Error::other(format!(
                    "PostgreSQL connection failed: {}",
                    e
                )))
            })?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("PostgreSQL connection error: {}", e);
            }
        });
        let store = Self {
            client: Arc::new(client),
            table: table.into(),
            metric: DistanceMetric::Cosine,
        };
        store.init_schema().await?;
        Ok(store)
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Drop the backing table. Useful for tests and ephemeral deployments.
    pub async fn drop_table(&self) -> Result<()> {
        let sql = format!("DROP TABLE IF EXISTS {}", quote_ident(&self.table));
        self.client
            .execute(&sql, &[])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Drop table failed: {}", e)))?;
        Ok(())
    }

    async fn init_schema(&self) -> Result<()> {
        self.client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Create vector extension failed: {}", e)))?;

        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}',
                embedding vector
            )",
            quote_ident(&self.table)
        );
        self.client
            .execute(&sql, &[])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Schema init failed: {}", e)))?;
        Ok(())
    }

    fn row_to_doc(row: &tokio_postgres::Row) -> Result<Document> {
        let id: String = row
            .try_get(0)
            .map_err(|e| RagError::VectorStoreError(format!("Row decode id failed: {}", e)))?;
        let content: String = row
            .try_get(1)
            .map_err(|e| RagError::VectorStoreError(format!("Row decode content failed: {}", e)))?;
        let metadata_json: String = row
            .try_get(2)
            .map_err(|e| RagError::VectorStoreError(format!("Row decode metadata failed: {}", e)))?;
        let metadata: HashMap<String, String> = serde_json::from_str::<serde_json::Value>(&metadata_json)
            .map_err(|e| RagError::VectorStoreError(format!("Metadata JSON parse failed: {}", e)))?
            .as_object()
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let embedding: Option<Vec<f32>> = row
            .try_get::<_, Option<Vector>>(3)
            .map_err(|e| RagError::VectorStoreError(format!("Row decode embedding failed: {}", e)))?
            .map(|v| v.to_vec());
        Ok(Document {
            id,
            content,
            metadata,
            embedding,
        })
    }

    fn metadata_filter_json(filter: &MetadataFilter) -> String {
        if filter.filters.is_empty() {
            return "{}".to_string();
        }
        let mut map = serde_json::Map::with_capacity(filter.filters.len());
        for (k, v) in &filter.filters {
            map.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        serde_json::Value::Object(map).to_string()
    }
}

fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

fn distance_operator(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "<=>",
        DistanceMetric::Euclidean => "<->",
        DistanceMetric::DotProduct => "<#>",
        DistanceMetric::Manhattan => "<+",
    }
}

fn distance_to_score(distance: f32, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::DotProduct => -distance,
        DistanceMetric::Euclidean | DistanceMetric::Manhattan => {
            if distance == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + distance)
            }
        }
    }
}

#[allow(async_fn_in_trait)]
impl VectorStore for PostgresVectorStore {
    async fn add(&self, document: Document) -> Result<()> {
        let sql = format!(
            "INSERT INTO {} (id, content, metadata, embedding)
             VALUES ($1, $2, $3::jsonb, $4)
             ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding",
            quote_ident(&self.table)
        );
        let metadata_json =
            serde_json::to_string(&document.metadata).map_err(RagError::JsonError)?;
        let embedding: Option<Vector> = document.embedding.map(Vector::from);
        self.client
            .execute(
                &sql,
                &[
                    &document.id,
                    &document.content,
                    &metadata_json,
                    &embedding,
                ],
            )
            .await
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
        let operator = distance_operator(self.metric);
        let sql = format!(
            "SELECT id, content, metadata, embedding,
                    (embedding {} $1) AS distance
             FROM {}
             WHERE embedding IS NOT NULL
               AND metadata @> $2::jsonb
             ORDER BY distance ASC
             LIMIT $3",
            operator,
            quote_ident(&self.table)
        );
        let query_vector = Vector::from(query.to_vec());
        let filter_json = Self::metadata_filter_json(filter);
        let rows = self
            .client
            .query(&sql, &[&query_vector, &filter_json, &(top_k as i64)])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Search failed: {}", e)))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let distance: f32 = row
                .try_get(4)
                .map_err(|e| RagError::VectorStoreError(format!("Distance decode failed: {}", e)))?;
            let document = Self::row_to_doc(&row)?;
            results.push(Similarity {
                document,
                score: distance_to_score(distance, self.metric),
            });
        }
        Ok(results)
    }

    async fn search_batch(&self, queries: &[Vec<f32>], top_k: usize) -> Result<Vec<Vec<Similarity>>> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.search(q, top_k).await?);
        }
        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<Document>> {
        let sql = format!(
            "SELECT id, content, metadata, embedding FROM {} WHERE id = $1",
            quote_ident(&self.table)
        );
        let row = self
            .client
            .query_opt(&sql, &[&id])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Get failed: {}", e)))?;
        match row {
            Some(r) => Ok(Some(Self::row_to_doc(&r)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let sql = format!("DELETE FROM {} WHERE id = $1", quote_ident(&self.table));
        let rows = self
            .client
            .execute(&sql, &[&id])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Delete failed: {}", e)))?;
        Ok(rows > 0)
    }

    async fn delete_batch(&self, ids: Vec<String>) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let sql = format!("DELETE FROM {} WHERE id = ANY($1)", quote_ident(&self.table));
        let rows = self
            .client
            .execute(&sql, &[&ids])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Delete batch failed: {}", e)))?;
        Ok(rows as usize)
    }

    async fn clear(&self) -> Result<()> {
        let sql = format!("DELETE FROM {}", quote_ident(&self.table));
        self.client
            .execute(&sql, &[])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Clear failed: {}", e)))?;
        Ok(())
    }

    async fn list(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        let sql = format!(
            "SELECT id, content, metadata, embedding FROM {}
             ORDER BY id
             LIMIT $1 OFFSET $2",
            quote_ident(&self.table)
        );
        let rows = self
            .client
            .query(&sql, &[&(limit as i64), &(offset as i64)])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("List failed: {}", e)))?;
        rows.iter().map(|r| Self::row_to_doc(r)).collect()
    }

    async fn count(&self) -> Result<usize> {
        let sql = format!("SELECT COUNT(*) FROM {}", quote_ident(&self.table));
        let row = self
            .client
            .query_one(&sql, &[])
            .await
            .map_err(|e| RagError::VectorStoreError(format!("Count failed: {}", e)))?;
        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }
}
