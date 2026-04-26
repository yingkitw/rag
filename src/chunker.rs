use crate::errors::Result;

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
}