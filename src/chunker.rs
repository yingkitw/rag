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