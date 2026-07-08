//! Append-only write-ahead log for incremental `VectorStore` updates.
//!
//! Instead of rewriting an entire JSON snapshot on every mutation, callers can
//! append a compact operation (`Put` / `Delete`) to a WAL file. On startup the
//! store replays the log to reconstruct state, then is free to checkpoint
//! (rewrite a fresh snapshot and truncate the log) at leisure. This keeps
//! incremental ingestion cheap and crash-safe.

use crate::errors::Result;
use crate::vector_store::Document;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// A single recorded mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    Put(Document),
    Delete(String),
}

/// Append-only write-ahead log backed by a newline-delimited JSON file.
pub struct WriteAheadLog {
    path: PathBuf,
}

impl WriteAheadLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a single operation, flushing immediately for durability.
    pub fn append(&self, op: &WalOp) -> Result<()> {
        let line = serde_json::to_string(op)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        file.sync_all()?;
        Ok(())
    }

    /// Append many operations in one write (single fsync).
    pub fn append_batch(&self, ops: &[WalOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for op in ops {
            let line = serde_json::to_string(op)?;
            writeln!(file, "{line}")?;
        }
        file.sync_all()?;
        Ok(())
    }

    /// Replay every recorded operation in order.
    pub fn replay(&self) -> Result<Vec<WalOp>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = OpenOptions::new().read(true).open(&self.path)?;
        let reader = BufReader::new(file);
        let mut ops = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WalOp>(&line) {
                Ok(op) => ops.push(op),
                Err(_) => continue, // tolerate a torn final line
            }
        }
        Ok(ops)
    }

    /// Truncate the log to zero length (call after a successful checkpoint).
    pub fn truncate(&self) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        file.sync_all()?;
        Ok(())
    }

    /// Number of bytes in the log file (0 if absent).
    pub fn size(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }
}

/// Apply a replayed log into a map keyed by document id. Later ops win.
pub fn apply_ops(
    ops: &[WalOp],
) -> std::collections::HashMap<String, Document> {
    let mut state: std::collections::HashMap<String, Document> = std::collections::HashMap::new();
    for op in ops {
        match op {
            WalOp::Put(doc) => {
                state.insert(doc.id.clone(), doc.clone());
            }
            WalOp::Delete(id) => {
                state.remove(id);
            }
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str) -> Document {
        Document::with_id(id.to_string(), format!("content-{id}"))
            .with_embedding(vec![1.0, 2.0])
    }

    #[test]
    fn test_append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.jsonl");
        let wal = WriteAheadLog::new(&path);
        wal.append(&WalOp::Put(doc("a"))).unwrap();
        wal.append(&WalOp::Put(doc("b"))).unwrap();
        wal.append(&WalOp::Delete("a".to_string())).unwrap();

        let ops = wal.replay().unwrap();
        assert_eq!(ops.len(), 3);
        let state = apply_ops(&ops);
        assert!(!state.contains_key("a"));
        assert!(state.contains_key("b"));
    }

    #[test]
    fn test_append_batch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.jsonl");
        let wal = WriteAheadLog::new(&path);
        let ops = vec![
            WalOp::Put(doc("1")),
            WalOp::Put(doc("2")),
            WalOp::Put(doc("3")),
        ];
        wal.append_batch(&ops).unwrap();
        assert_eq!(wal.replay().unwrap().len(), 3);
    }

    #[test]
    fn test_append_batch_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        let wal = WriteAheadLog::new(&path);
        wal.append_batch(&[]).unwrap();
    }

    #[test]
    fn test_replay_missing_file() {
        let wal = WriteAheadLog::new("/nonexistent/wal.jsonl");
        let ops = wal.replay().unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc.jsonl");
        let wal = WriteAheadLog::new(&path);
        wal.append(&WalOp::Put(doc("a"))).unwrap();
        assert!(wal.size() > 0);
        wal.truncate().unwrap();
        assert_eq!(wal.size(), 0);
        assert_eq!(wal.replay().unwrap().len(), 0);
    }

    #[test]
    fn test_torn_line_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("torn.jsonl");
        let wal = WriteAheadLog::new(&path);
        wal.append(&WalOp::Put(doc("a"))).unwrap();
        // Append a partial (invalid) line to simulate a torn write.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(f, "{{partial").unwrap();
        }
        let ops = wal.replay().unwrap();
        // Valid op still recovered; torn line skipped.
        assert!(ops.iter().any(|o| matches!(o, WalOp::Put(d) if d.id == "a")));
    }
}
