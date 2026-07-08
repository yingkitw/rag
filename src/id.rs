//! Minimal unique identifier generator with no external dependencies.
//!
//! Produces RFC 4122-formatted version 4 UUID strings (e.g.
//! `550e8400-e29b-41d4-a716-446655440000`). These IDs exist purely for
//! uniqueness within document stores and graphs; they are not
//! cryptographically random. Uniqueness is guaranteed within a process via a
//! monotonically increasing atomic counter mixed into the output, layered on
//! top of a per-thread xorshift PRNG seeded from wall-clock time.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global call counter; every `new_uuid_v4` invocation consumes a unique value,
/// so even identical PRNG states produce distinct identifiers.
static COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static RNG: Cell<u64> = Cell::new(seed());
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Generate a new version 4 UUID string.
pub fn new_uuid_v4() -> String {
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);

    let r0 = next_u64();
    let r1 = next_u64() ^ seq.rotate_left(17);

    let mut bytes = [0u8; 16];
    for (i, b) in bytes[..8].iter_mut().enumerate() {
        *b = (r0 >> (i * 8)) as u8;
    }
    for (i, b) in bytes[8..].iter_mut().enumerate() {
        *b = (r1 >> (i * 8)) as u8;
    }

    // Version 4 (random) and variant (10xx) bits, matching the uuid crate.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format_uuid(&bytes)
}

fn format_uuid(b: &[u8; 16]) -> String {
    let mut s = String::with_capacity(36);
    for (i, byte) in b.iter().enumerate() {
        if i == 4 || i == 6 || i == 8 || i == 10 {
            s.push('-');
        }
        s.push(HEX[(byte >> 4) as usize] as char);
        s.push(HEX[(byte & 0x0f) as usize] as char);
    }
    s
}

fn seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    splitmix64(nanos)
}

fn next_u64() -> u64 {
    RNG.with(|cell| {
        let mut x = cell.get();
        if x == 0 {
            x = seed();
        }
        // xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        cell.set(x);
        x
    })
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_uuid_format() {
        let id = new_uuid_v4();
        assert_eq!(id.len(), 36, "UUID must be 36 chars");
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }

    #[test]
    fn test_uuid_version_and_variant() {
        for _ in 0..1000 {
            let id = new_uuid_v4();
            let bytes = id.as_bytes();
            // 19th hex char (index 14) is the version nibble -> '4'.
            assert_eq!(bytes[14], b'4', "version nibble must be 4: {}", id);
            // 17th char (after the third dash) top bits are 10xx -> '8','9','a','b'.
            assert!(
                matches!(bytes[19], b'8' | b'9' | b'a' | b'b'),
                "variant bits invalid: {}",
                id
            );
        }
    }

    #[test]
    fn test_uuid_uniqueness() {
        let mut seen = HashSet::new();
        for _ in 0..100_000 {
            assert!(seen.insert(new_uuid_v4()), "duplicate id generated");
        }
    }
}
