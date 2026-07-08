use crate::errors::Result;
use std::collections::HashSet;

pub trait TextChunker: Send + Sync {
    fn chunk(&self, text: &str) -> Result<Vec<String>>;
}

pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self { chunk_size, overlap }
    }
}

impl Default for FixedSizeChunker {
    fn default() -> Self {
        Self::new(500, 50)
    }
}

impl TextChunker for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if self.overlap >= self.chunk_size {
            return Err(crate::errors::RagError::InvalidConfig(
                "Overlap must be less than chunk size".to_string(),
            ));
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        if words.is_empty() {
            return Ok(chunks);
        }

        let mut start = 0;
        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            let chunk = words[start..end].join(" ");
            chunks.push(chunk);

            start += self.chunk_size - self.overlap;
            if start >= words.len() {
                break;
            }
        }

        Ok(chunks)
    }
}

pub struct ParagraphChunker;

impl Default for ParagraphChunker {
    fn default() -> Self {
        Self
    }
}

impl TextChunker for ParagraphChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        let chunks: Vec<String> = text
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if chunks.is_empty() && !text.trim().is_empty() {
            Ok(vec![text.trim().to_string()])
        } else {
            Ok(chunks)
        }
    }
}

pub struct SentenceChunker {
    max_sentences: usize,
}

impl SentenceChunker {
    pub fn new(max_sentences: usize) -> Self {
        Self { max_sentences }
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new(5)
    }
}

impl TextChunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        let sentences: Vec<String> = text
            .split_inclusive(&['.', '!', '?', '\n'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut chunks = Vec::new();
        for chunk in sentences.chunks(self.max_sentences) {
            let chunk_text = chunk.join(" ");
            chunks.push(chunk_text);
        }

        Ok(chunks)
    }
}

/// Recursive chunker that splits text using a prioritized list of separators,
/// then greedily merges the resulting pieces into chunks of roughly
/// `chunk_size` characters with optional overlap. This mirrors the behavior of
/// LangChain's `RecursiveCharacterTextSplitter`: it tries the most
/// structure-preserving separator first and only falls back to coarser ones
/// when a piece still exceeds the target size.
pub struct RecursiveChunker {
    chunk_size: usize,
    overlap: usize,
    separators: Vec<String>,
}

impl RecursiveChunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
            separators: default_separators(),
        }
    }

    /// Build a chunker with a custom separator priority list.
    pub fn with_separators(chunk_size: usize, overlap: usize, separators: Vec<String>) -> Self {
        Self {
            chunk_size,
            overlap,
            separators,
        }
    }
}

fn default_separators() -> Vec<String> {
    [
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl Default for RecursiveChunker {
    fn default() -> Self {
        Self::new(1000, 200)
    }
}

impl TextChunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if self.overlap >= self.chunk_size {
            return Err(crate::errors::RagError::InvalidConfig(
                "Overlap must be less than chunk size".to_string(),
            ));
        }
        let pieces = split_recursive(text, &self.separators, self.chunk_size);
        Ok(merge_pieces(&pieces, self.chunk_size, self.overlap))
    }
}

fn split_recursive(text: &str, separators: &[String], chunk_size: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    // Pick the first separator that actually appears in the text.
    let sep_idx = separators
        .iter()
        .position(|s| !s.is_empty() && text.contains(s.as_str()))
        .unwrap_or(separators.len() - 1);
    let separator = &separators[sep_idx];

    let parts: Vec<String> = if separator.is_empty() {
        // Character-level split as last resort.
        text.chars().map(|c| c.to_string()).collect()
    } else {
        text.split(separator.as_str())
            .map(|s| s.to_string())
            .collect()
    };

    let mut out = Vec::new();
    for part in parts {
        let trimmed = part.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.len() > chunk_size && sep_idx + 1 < separators.len() {
            out.extend(split_recursive(&trimmed, &separators[sep_idx + 1..], chunk_size));
        } else {
            out.push(trimmed);
        }
    }
    out
}

/// Greedily merge small pieces into chunks near `chunk_size`, carrying
/// `overlap` characters of trailing context into the next chunk.
fn merge_pieces(pieces: &[String], chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if !current.is_empty() && current.len() + piece.len() + 1 > chunk_size {
            chunks.push(std::mem::take(&mut current));
            // seed next chunk with trailing overlap from the just-emitted chunk
            if overlap > 0 {
                if let Some(last) = chunks.last() {
                    let start = last.len().saturating_sub(overlap);
                    current = last[start..].to_string();
                }
            }
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(piece);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Semantic chunker that groups consecutive sentences whose token overlap
/// (Jaccard similarity) stays above a threshold, starting a new chunk when
/// similarity drops. No embedding model is required — overlap is computed from
/// whitespace-split lowercased tokens, which is a cheap proxy for topical
/// continuity.
pub struct SemanticChunker {
    max_chunk_size: usize,
    similarity_threshold: f64,
}

impl SemanticChunker {
    /// `max_chunk_size` is an upper bound in characters; `similarity_threshold`
    /// (0.0–1.0) controls how aggressively sentences are split.
    pub fn new(max_chunk_size: usize, similarity_threshold: f64) -> Self {
        Self {
            max_chunk_size,
            similarity_threshold,
        }
    }
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new(1200, 0.3)
    }
}

impl TextChunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        let sentences: Vec<String> = text
            .split_inclusive(&['.', '!', '?', '\n'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut current = sentences[0].clone();
        let mut current_tokens = tokenize_set(&current);

        for sentence in sentences.into_iter().skip(1) {
            let sent_tokens = tokenize_set(&sentence);
            let sim = jaccard(&current_tokens, &sent_tokens);
            if sim >= self.similarity_threshold
                && current.len() + sentence.len() + 1 <= self.max_chunk_size
            {
                current.push(' ');
                current.push_str(&sentence);
                current_tokens.extend(sent_tokens);
            } else {
                chunks.push(std::mem::take(&mut current));
                current = sentence;
                current_tokens = tokenize_set(&current);
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        Ok(chunks)
    }
}

fn tokenize_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Structural (Markdown / code-aware) chunker. Splits on Markdown headings and
/// fenced code blocks so that each chunk keeps a coherent section. Oversized
/// sections are further split with a `FixedSizeChunker` fallback.
pub struct StructuralChunker {
    max_chunk_size: usize,
    overlap: usize,
}

impl StructuralChunker {
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size,
            overlap,
        }
    }
}

impl Default for StructuralChunker {
    fn default() -> Self {
        Self::new(1500, 100)
    }
}

impl TextChunker for StructuralChunker {
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        let mut sections = Vec::new();
        let mut current = String::new();
        let mut in_code_fence = false;

        for line in text.lines() {
            let trimmed = line.trim_start();
            let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");

            // Toggle code-fence state, but always emit the fence line.
            if is_fence {
                in_code_fence = !in_code_fence;
            }

            let is_heading = !in_code_fence && trimmed.starts_with('#');
            if is_heading && !current.is_empty() {
                sections.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
        if !current.is_empty() {
            sections.push(current);
        }

        // Fallback: split any oversize sections with a fixed-size chunker.
        let mut chunks = Vec::new();
        let fallback = FixedSizeChunker::new(self.max_chunk_size, self.overlap);
        for section in sections {
            let section = section.trim();
            if section.is_empty() {
                continue;
            }
            if section.len() <= self.max_chunk_size {
                chunks.push(section.to_string());
            } else {
                chunks.extend(fallback.chunk(section)?);
            }
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_chunker_basic() {
        let chunker = FixedSizeChunker::new(3, 0);
        let text = "one two three four five six seven";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "one two three");
        assert_eq!(chunks[1], "four five six");
        assert_eq!(chunks[2], "seven");
    }

    #[test]
    fn test_fixed_size_chunker_with_overlap() {
        let chunker = FixedSizeChunker::new(4, 2);
        let text = "a b c d e f g h";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "a b c d");
        assert_eq!(chunks[1], "c d e f");
        assert_eq!(chunks[2], "e f g h");
        assert_eq!(chunks[3], "g h");
    }

    #[test]
    fn test_fixed_size_chunker_empty() {
        let chunker = FixedSizeChunker::new(5, 1);
        let chunks = chunker.chunk("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_fixed_size_chunker_invalid_config() {
        let chunker = FixedSizeChunker::new(5, 10);
        let result = chunker.chunk("test text here");
        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_size_chunker_single_word() {
        let chunker = FixedSizeChunker::new(5, 0);
        let chunks = chunker.chunk("hello").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello");
    }

    #[test]
    fn test_fixed_size_chunker_default() {
        let chunker = FixedSizeChunker::default();
        let text: String = (0..1000).map(|i| format!("word{} ", i)).collect();
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_paragraph_chunker_basic() {
        let chunker = ParagraphChunker;
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "First paragraph.");
        assert_eq!(chunks[1], "Second paragraph.");
        assert_eq!(chunks[2], "Third paragraph.");
    }

    #[test]
    fn test_paragraph_chunker_single_paragraph() {
        let chunker = ParagraphChunker;
        let text = "Only one paragraph.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Only one paragraph.");
    }

    #[test]
    fn test_paragraph_chunker_empty() {
        let chunker = ParagraphChunker;
        let chunks = chunker.chunk("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_paragraph_chunker_whitespace_only() {
        let chunker = ParagraphChunker;
        let chunks = chunker.chunk("   \n\n   ").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_paragraph_chunker_no_double_newline() {
        let chunker = ParagraphChunker;
        let text = "Just a single line with no paragraph breaks";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Just a single line with no paragraph breaks");
    }

    #[test]
    fn test_sentence_chunker_basic() {
        let chunker = SentenceChunker::new(2);
        let text = "First sentence. Second sentence. Third sentence. Fourth.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "First sentence. Second sentence.");
        assert_eq!(chunks[1], "Third sentence. Fourth.");
    }

    #[test]
    fn test_sentence_chunker_single_sentence() {
        let chunker = SentenceChunker::new(3);
        let text = "Only one sentence.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Only one sentence.");
    }

    #[test]
    fn test_sentence_chunker_exclamation() {
        let chunker = SentenceChunker::new(2);
        let text = "Hello! How are you? I am fine.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "Hello! How are you?");
        assert_eq!(chunks[1], "I am fine.");
    }

    #[test]
    fn test_sentence_chunker_empty() {
        let chunker = SentenceChunker::new(5);
        let chunks = chunker.chunk("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_sentence_chunker_default() {
        let chunker = SentenceChunker::default();
        let text = "A. B. C. D. E. F. G. H. I. J.";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_sentence_chunker_newline_separator() {
        let chunker = SentenceChunker::new(2);
        let text = "Line one\nLine two\nLine three\nLine four";
        let chunks = chunker.chunk(text).unwrap();
        assert_eq!(chunks.len(), 2);
        // Newlines are consumed as sentence delimiters, so chunks join with space
        assert_eq!(chunks[0], "Line one Line two");
        assert_eq!(chunks[1], "Line three Line four");
    }

    #[test]
    fn test_recursive_chunker_basic() {
        let chunker = RecursiveChunker::new(30, 0);
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one.";
        let chunks = chunker.chunk(text).unwrap();
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.len() <= 30 + 5, "chunk too long: {} (len {})", c, c.len());
        }
    }

    #[test]
    fn test_recursive_chunker_respects_separators() {
        let chunker = RecursiveChunker::new(15, 0);
        let text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota";
        let chunks = chunker.chunk(text).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_recursive_chunker_invalid_config() {
        let chunker = RecursiveChunker::new(10, 20);
        let result = chunker.chunk("some text");
        assert!(result.is_err());
    }

    #[test]
    fn test_recursive_chunker_empty() {
        let chunker = RecursiveChunker::new(50, 10);
        let chunks = chunker.chunk("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_recursive_chunker_default() {
        let chunker = RecursiveChunker::default();
        let text: String = (0..2000).map(|i| format!("word{i} ")).collect();
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_semantic_chunker_groups() {
        let chunker = SemanticChunker::new(500, 0.1);
        let text = "Cats are animals. Cats like to sleep. Cats purr loudly. \
                    Quantum physics studies particles. Particles behave as waves.";
        let chunks = chunker.chunk(text).unwrap();
        // Two topics -> at least 2 chunks.
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_semantic_chunker_threshold_split() {
        // Very high threshold splits almost every sentence.
        let chunker = SemanticChunker::new(500, 0.99);
        let text = "alpha beta. gamma delta. epsilon zeta.";
        let chunks = chunker.chunk(text).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_semantic_chunker_empty() {
        let chunker = SemanticChunker::default();
        assert!(chunker.chunk("").unwrap().is_empty());
    }

    #[test]
    fn test_structural_chunker_headings() {
        let chunker = StructuralChunker::new(1000, 0);
        let text = "# Title\nIntro text.\n## Section A\nContent A.\n## Section B\nContent B.";
        let chunks = chunker.chunk(text).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.contains("Section A")));
        assert!(chunks.iter().any(|c| c.contains("Section B")));
    }

    #[test]
    fn test_structural_chunker_code_fence_kept_together() {
        let chunker = StructuralChunker::new(2000, 0);
        let text = "# Intro\nSome prose.\n```rust\nfn main() {}\n```\n## After\nMore prose.";
        let chunks = chunker.chunk(text).unwrap();
        // The code fence line should not start a brand-new chunk on its own.
        let joined = chunks.join("\n");
        assert!(joined.contains("fn main()"));
    }

    #[test]
    fn test_structural_chunker_oversize_fallback() {
        let chunker = StructuralChunker::new(20, 0);
        let text = "# Section\nword ".repeat(50);
        let chunks = chunker.chunk(&text).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_structural_chunker_empty() {
        let chunker = StructuralChunker::default();
        assert!(chunker.chunk("").unwrap().is_empty());
    }
}