//! Example: Compressed JSON persistence for snapshots (requires `compress` feature).

#[cfg(feature = "compress")]
use rag::compress::{compress_bytes, save_compressed};

#[cfg(feature = "compress")]
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Snapshot {
    version: u32,
    docs: Vec<String>,
}

#[cfg(feature = "compress")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snap = Snapshot {
        version: 1,
        docs: (0..1000).map(|i| format!("document number {i}")).collect(),
    };

    let dir = std::env::temp_dir().join("rag_compress_example");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("snapshot.zst");

    // Serialize + compress + write.
    save_compressed(&snap, &path, 19)?;
    let json = serde_json::to_vec_pretty(&snap)?;
    let compressed = compress_bytes(&json, 19)?;
    println!("raw json bytes   : {}", json.len());
    println!("compressed bytes : {} ({:.1}x smaller)", compressed.len(), json.len() as f64 / compressed.len() as f64);
    println!("wrote snapshot to {}", path.display());
    Ok(())
}

#[cfg(not(feature = "compress"))]
fn main() {
    eprintln!("Rebuild with: cargo run --example compressed_persistence --features compress");
}
