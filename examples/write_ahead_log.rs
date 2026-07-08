//! Example: Append-only write-ahead log for incremental vector-store updates.

use rag::vector_store::Document;
use rag::wal::{WalOp, WriteAheadLog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("rag_wal_example.jsonl");
    let wal = WriteAheadLog::new(&path);

    // Append some incremental mutations.
    wal.append(&WalOp::Put(Document::with_id(
        "doc-1".to_string(),
        "first chunk".to_string(),
    )))?;
    wal.append(&WalOp::Put(Document::with_id(
        "doc-2".to_string(),
        "second chunk".to_string(),
    )))?;
    wal.append_batch(&[
        WalOp::Put(Document::with_id(
            "doc-3".to_string(),
            "third chunk".to_string(),
        )),
        WalOp::Delete("doc-1".to_string()),
    ])?;

    println!("WAL size on disk: {} bytes", wal.size());

    // Replay to reconstruct state.
    let ops = wal.replay()?;
    let state = rag::wal::apply_ops(&ops);
    println!("Recovered {} documents after replay:", state.len());
    for (id, doc) in &state {
        println!("  {id}: {:?}", doc.content);
    }

    // After a checkpoint (rewrite a fresh snapshot), truncate the log.
    wal.truncate()?;
    println!("Truncated. WAL size now: {} bytes", wal.size());
    Ok(())
}
