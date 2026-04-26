use crate::errors::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A document extracted from an external source, ready for chunking and embedding.
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    pub content: String,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

impl ExtractedDocument {
    pub fn new(content: String, source: String) -> Self {
        Self {
            content,
            source,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Trait for text extraction from various document sources.
#[allow(async_fn_in_trait)]
pub trait Source: Send + Sync {
    /// Extract documents from the source.
    async fn extract(&self) -> Result<Vec<ExtractedDocument>>;
}

// ============================================================
// PDF Source
// ============================================================

/// Extract text from PDF files using `lopdf`.
pub struct PdfSource {
    path: PathBuf,
}

impl PdfSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl Source for PdfSource {
    async fn extract(&self) -> Result<Vec<ExtractedDocument>> {
        let doc = lopdf::Document::load(&self.path).map_err(|e| {
            crate::errors::RagError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to load PDF: {}", e),
            ))
        })?;

        let mut content = String::new();
        let pages = doc.get_pages();

        for (page_num, page_id) in pages {
            if let Ok(lopdf::Object::Dictionary(page)) = doc.get_object(page_id) {
                if let Ok(text) = extract_text_from_page(&doc, page) {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&format!("--- Page {} ---\n", page_num));
                    content.push_str(&text);
                }
            }
        }

        let source_name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf")
            .to_string();

        let doc = ExtractedDocument::new(content, source_name)
            .with_metadata("format".to_string(), "pdf".to_string())
            .with_metadata(
                "path".to_string(),
                self.path.to_string_lossy().to_string(),
            );

        Ok(vec![doc])
    }
}

/// Extract text strings from a PDF page by walking content streams.
fn extract_text_from_page(doc: &lopdf::Document, page: &lopdf::Dictionary) -> Result<String> {
    let mut text = String::new();

    if let Ok(contents) = page.get(b"Contents") {
        let streams = match contents {
            lopdf::Object::Reference(ref_id) => {
                vec![doc.get_object(*ref_id).cloned().unwrap_or(lopdf::Object::Null)]
            }
            lopdf::Object::Array(arr) => arr.clone(),
            other => vec![other.clone()],
        };

        for stream_obj in streams {
            if let Ok(stream_ref) = stream_obj.as_stream() {
                let mut stream = stream_ref.clone();
                stream.decompress();
                let extracted = extract_text_from_content(&stream.content);
                text.push_str(&extracted);
            }
        }
    }

    Ok(text)
}

/// Naive text extraction from raw PDF content stream bytes.
/// Looks for text operators (TJ, Tj) and extracts string operands.
fn extract_text_from_content(data: &[u8]) -> String {
    let mut text = String::new();
    let mut i = 0;

    while i < data.len() {
        // Look for text-showing operators
        if i + 2 < data.len() && &data[i..i + 2] == b"Tj" {
            // Tj operator: preceding string on stack
            // Walk backward to find the string
            if let Some((start, end)) = find_preceding_string(data, i) {
                if let Ok(s) = decode_pdf_string(&data[start..end]) {
                    text.push_str(&s);
                }
            }
            i += 2;
        } else if i + 2 < data.len() && &data[i..i + 2] == b"TJ" {
            // TJ operator: array of strings and numbers
            if let Some((start, end)) = find_preceding_array(data, i) {
                text.push_str(&extract_strings_from_array(&data[start..end]));
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    text
}

/// Find a literal string `(...)` immediately before position `pos`.
fn find_preceding_string(data: &[u8], pos: usize) -> Option<(usize, usize)> {
    let mut depth = 0;
    let mut end = pos;
    // Skip whitespace backward
    while end > 0 && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    if end == 0 || data[end - 1] != b')' {
        return None;
    }

    let mut start = end - 1;
    while start > 0 {
        match data[start] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start + 1, end - 1));
                }
            }
            b'\\' if start + 1 < data.len() => {
                // escaped char, skip next
                if start > 0 {
                    start -= 1;
                }
            }
            _ => {}
        }
        if start == 0 {
            break;
        }
        start -= 1;
    }
    None
}

/// Find an array `[...]` immediately before position `pos`.
fn find_preceding_array(data: &[u8], pos: usize) -> Option<(usize, usize)> {
    let mut end = pos;
    while end > 0 && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    if end == 0 || data[end - 1] != b']' {
        return None;
    }

    let mut depth = 1;
    let mut start = end - 1;
    while start > 0 {
        start -= 1;
        match data[start] {
            b']' => depth += 1,
            b'[' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start + 1, end - 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract all literal strings from a PDF array slice.
fn extract_strings_from_array(data: &[u8]) -> String {
    let mut result = String::new();
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'(' {
            if let Some(end) = find_matching_paren(data, i) {
                if let Ok(s) = decode_pdf_string(&data[i + 1..end]) {
                    result.push_str(&s);
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }
    result
}

/// Find the matching `)` for a `(` at position `start`.
fn find_matching_paren(data: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut i = start + 1;
    while i < data.len() {
        match data[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            b'\\' => i += 1, // skip escaped char
            _ => {}
        }
        i += 1;
    }
    None
}

/// Decode a PDF string (handling common escapes).
fn decode_pdf_string(data: &[u8]) -> Result<String> {
    let mut result = String::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\\' && i + 1 < data.len() {
            match data[i + 1] {
                b'n' => result.push('\n'),
                b'r' => result.push('\r'),
                b't' => result.push('\t'),
                b'b' => result.push('\x08'),
                b'f' => result.push('\x0C'),
                b'(' => result.push('('),
                b')' => result.push(')'),
                b'\\' => result.push('\\'),
                b'0'..=b'9' => {
                    // Octal escape
                    let mut octal = String::new();
                    for j in 0..3 {
                        if i + 1 + j < data.len() && data[i + 1 + j].is_ascii_digit() {
                            octal.push(data[i + 1 + j] as char);
                        } else {
                            break;
                        }
                    }
                    if let Ok(val) = u32::from_str_radix(&octal, 8) {
                        if let Some(c) = char::from_u32(val) {
                            result.push(c);
                        }
                    }
                    i += octal.len().saturating_sub(1);
                }
                _ => result.push(data[i + 1] as char),
            }
            i += 2;
        } else {
            result.push(data[i] as char);
            i += 1;
        }
    }
    Ok(result)
}

// ============================================================
// Codebase Source
// ============================================================

/// Extract text from source code files in a directory tree.
pub struct CodebaseSource {
    root: PathBuf,
    extensions: Vec<String>,
    max_file_size: usize,
}

impl CodebaseSource {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            extensions: vec![
                "rs", "py", "js", "ts", "java", "go", "cpp", "c", "h", "hpp",
                "rb", "php", "swift", "kt", "scala", "r", "md", "txt", "toml",
                "yaml", "yml", "json", "xml", "html", "css", "sh", "bash",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            max_file_size: 1024 * 1024, // 1 MB
        }
    }

    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn with_max_file_size(mut self, bytes: usize) -> Self {
        self.max_file_size = bytes;
        self
    }
}

impl Source for CodebaseSource {
    async fn extract(&self) -> Result<Vec<ExtractedDocument>> {
        let mut docs = Vec::new();

        let entries = walkdir::WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in entries {
            let path = entry.path();

            // Skip hidden directories and common build artifacts
            let path_str = path.to_string_lossy();
            let skip_patterns = [
                "/.git/", "/.github/", "/.vscode/", "/.idea/", "/.cargo/",
                "/target/", "/node_modules/", "/vendor/", "/dist/", "/build/",
                "/__pycache__/", "/.mypy_cache/", "/.pytest_cache/",
            ];
            if skip_patterns.iter().any(|p| path_str.contains(p)) {
                continue;
            }
            // Skip hidden files (names starting with dot)
            if path.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with('.'))
            {
                continue;
            }

            // Check extension
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !self.extensions.iter().any(|e| e == ext) {
                continue;
            }

            // Check file size
            let metadata = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if metadata.len() as usize > self.max_file_size {
                continue;
            }

            // Read content
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue, // Skip binary files
            };

            let relative = path.strip_prefix(&self.root).unwrap_or(path);
            let doc = ExtractedDocument::new(content, relative.to_string_lossy().to_string())
                .with_metadata("format".to_string(), "code".to_string())
                .with_metadata("extension".to_string(), ext.to_string())
                .with_metadata(
                    "path".to_string(),
                    path.to_string_lossy().to_string(),
                );

            docs.push(doc);
        }

        Ok(docs)
    }
}

// ============================================================
// Wiki Source
// ============================================================

/// Extract text from Wikipedia pages via the REST API.
pub struct WikiSource {
    title: String,
    language: String,
}

impl WikiSource {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            language: "en".to_string(),
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }
}

impl Source for WikiSource {
    async fn extract(&self) -> Result<Vec<ExtractedDocument>> {
        let url = format!(
            "https://{}.wikipedia.org/api/rest_v1/page/summary/{}",
            self.language,
            urlencoding::encode(&self.title.replace(' ', "_"))
        );

        let response = reqwest::get(&url).await.map_err(|e| {
            crate::errors::RagError::HttpError(e)
        })?;

        if !response.status().is_success() {
            return Err(crate::errors::RagError::EmbeddingError(format!(
                "Wiki API returned status: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response.json().await.map_err(|e| {
            crate::errors::RagError::HttpError(e)
        })?;

        let title = body["title"]
            .as_str()
            .unwrap_or(&self.title)
            .to_string();

        let extract = body["extract"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if extract.is_empty() {
            return Err(crate::errors::RagError::EmbeddingError(
                "Wiki page has no extractable content".to_string(),
            ));
        }

        let doc = ExtractedDocument::new(extract, title.clone())
            .with_metadata("format".to_string(), "wiki".to_string())
            .with_metadata("language".to_string(), self.language.clone())
            .with_metadata(
                "url".to_string(),
                format!(
                    "https://{}.wikipedia.org/wiki/{}",
                    self.language,
                    urlencoding::encode(&title.replace(' ', "_"))
                ),
            );

        Ok(vec![doc])
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_document_new() {
        let doc = ExtractedDocument::new("hello".to_string(), "test.txt".to_string());
        assert_eq!(doc.content, "hello");
        assert_eq!(doc.source, "test.txt");
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_extracted_document_with_metadata() {
        let doc = ExtractedDocument::new("content".to_string(), "src.rs".to_string())
            .with_metadata("lang".to_string(), "rust".to_string());
        assert_eq!(doc.metadata.get("lang"), Some(&"rust".to_string()));
    }

    #[test]
    fn test_codebase_source_new() {
        let src = CodebaseSource::new("/tmp/test");
        assert_eq!(src.root, PathBuf::from("/tmp/test"));
        assert!(!src.extensions.is_empty());
        assert_eq!(src.max_file_size, 1024 * 1024);
    }

    #[test]
    fn test_codebase_source_with_extensions() {
        let src = CodebaseSource::new("/tmp/test").with_extensions(vec!["rs".to_string()]);
        assert_eq!(src.extensions, vec!["rs".to_string()]);
    }

    #[test]
    fn test_codebase_source_with_max_file_size() {
        let src = CodebaseSource::new("/tmp/test").with_max_file_size(512);
        assert_eq!(src.max_file_size, 512);
    }

    #[test]
    fn test_wiki_source_new() {
        let src = WikiSource::new("Rust (programming language)");
        assert_eq!(src.title, "Rust (programming language)");
        assert_eq!(src.language, "en");
    }

    #[test]
    fn test_wiki_source_with_language() {
        let src = WikiSource::new("Rust").with_language("ja");
        assert_eq!(src.language, "ja");
    }

    #[test]
    fn test_pdf_source_new() {
        let src = PdfSource::new("/tmp/test.pdf");
        assert_eq!(src.path, PathBuf::from("/tmp/test.pdf"));
    }

    #[test]
    fn test_decode_pdf_string_simple() {
        let data = b"hello world";
        let result = decode_pdf_string(data).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_decode_pdf_string_with_escapes() {
        let data = b"hello\\nworld";
        let result = decode_pdf_string(data).unwrap();
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_find_matching_paren() {
        let data = b"(hello world)";
        assert_eq!(find_matching_paren(data, 0), Some(12));
    }

    #[test]
    fn test_find_matching_paren_nested() {
        let data = b"(a (b) c)";
        assert_eq!(find_matching_paren(data, 0), Some(8));
    }

    #[test]
    fn test_extract_strings_from_array() {
        let data = b"(hello) 123 (world)";
        let result = extract_strings_from_array(data);
        assert_eq!(result, "helloworld");
    }

    #[tokio::test]
    async fn test_codebase_source_extracts_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();
        std::fs::write(root.join("readme.md"), "# My Project\n\nHello world.").unwrap();
        std::fs::write(root.join("config.toml"), "[package]\nname = \"test\"").unwrap();

        let src = CodebaseSource::new(root).with_extensions(vec![
            "rs".to_string(),
            "md".to_string(),
            "toml".to_string(),
        ]);

        let docs = src.extract().await.unwrap();
        assert_eq!(docs.len(), 4);

        let contents: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        assert!(contents.iter().any(|c| c.contains("fn main()")));
        assert!(contents.iter().any(|c| c.contains("pub fn add")));
        assert!(contents.iter().any(|c| c.contains("# My Project")));
        assert!(contents.iter().any(|c| c.contains("[package]")));
    }

    #[tokio::test]
    async fn test_codebase_source_skips_hidden_and_build_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        std::fs::create_dir(root.join("target")).unwrap();
        std::fs::write(root.join("target").join("debug"), "binary").unwrap();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(root.join(".git").join("config"), "[core]").unwrap();

        let src = CodebaseSource::new(root);
        let docs = src.extract().await.unwrap();

        // Should only find main.rs
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "main.rs");
    }

    #[tokio::test]
    async fn test_codebase_source_respects_max_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("small.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("large.rs"), "x".repeat(2000)).unwrap();

        let src = CodebaseSource::new(root)
            .with_extensions(vec!["rs".to_string()])
            .with_max_file_size(1000);

        let docs = src.extract().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "small.rs");
    }

    #[tokio::test]
    async fn test_codebase_source_ignores_unmatched_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("code.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("data.json"), "{\"key\": \"value\"}").unwrap();

        let src = CodebaseSource::new(root).with_extensions(vec!["rs".to_string()]);
        let docs = src.extract().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "code.rs");
    }
}
