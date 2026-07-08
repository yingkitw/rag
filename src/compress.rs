//! Compressed JSON persistence helpers.
//!
//! When the `compress` feature is enabled, `save_compressed` and
//! `load_compressed` serialize a value to JSON and write it through a zstd
//! compressor (level 19 by default), shrinking `vectors.json` / `graph.json`
//! style snapshots dramatically. The on-disk format is a raw zstd frame, which
//! is easily inspected with the `zstd` CLI.

use crate::errors::Result;
use serde::{de::DeserializeOwned, Serialize};

#[cfg(feature = "compress")]
use std::io::{Read, Write};

/// Serialize `value` to pretty JSON, compress with zstd, and write to `path`.
#[cfg(feature = "compress")]
pub fn save_compressed<T: Serialize>(value: &T, path: &std::path::Path, level: i32) -> Result<()> {
    let json = serde_json::to_vec_pretty(value)?;
    let file = std::fs::File::create(path)?;
    let mut encoder = zstd::stream::Encoder::new(file, level)?;
    encoder.write_all(&json)?;
    encoder.finish()?;
    Ok(())
}

/// Read a zstd-compressed JSON file and deserialize into `T`.
#[cfg(feature = "compress")]
pub fn load_compressed<T: DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let file = std::fs::File::open(path)?;
    let mut decoder = zstd::stream::Decoder::new(file)?;
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok(value)
}

/// Compress an arbitrary byte buffer in memory.
#[cfg(feature = "compress")]
pub fn compress_bytes(data: &[u8], level: i32) -> Result<Vec<u8>> {
    Ok(zstd::encode_all(data, level)?)
}

/// Decompress a zstd byte buffer in memory.
#[cfg(feature = "compress")]
pub fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::decode_all(data)?)
}

#[cfg(not(feature = "compress"))]
compile_error!(
    "the `compress` feature must be enabled to use rag::compress::{save_compressed, load_compressed}"
);

#[cfg(all(test, feature = "compress"))]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Data {
        name: String,
        values: Vec<f32>,
    }

    #[test]
    fn test_roundtrip_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.zst");
        let data = Data {
            name: "test".to_string(),
            values: vec![0.1; 1000],
        };
        save_compressed(&data, &path, 3).unwrap();
        let back: Data = load_compressed(&path).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn test_compress_decompress_bytes() {
        let input = b"hello world".repeat(1000);
        let compressed = compress_bytes(&input, 9).unwrap();
        assert!(compressed.len() < input.len());
        let back = decompress_bytes(&compressed).unwrap();
        assert_eq!(back, input);
    }

    #[test]
    fn test_compressed_smaller_than_json() {
        let data = Data {
            name: "repeating".to_string(),
            values: vec![0.5; 10_000],
        };
        let json = serde_json::to_vec_pretty(&data).unwrap();
        let compressed = compress_bytes(&json, 19).unwrap();
        assert!(compressed.len() < json.len());
    }
}
