//! Levenshtein edit distance for fuzzy string matching.

/// Compute Levenshtein distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }
    let mut prev = (0..=b_len).collect::<Vec<_>>();
    let mut curr = vec![0; b_len + 1];
    for (i, ac) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, bc) in b.chars().enumerate() {
            let cost = if ac == bc { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Check if `term` is within `max_distance` edits of `query`.
pub fn is_fuzzy_match(query: &str, term: &str, max_distance: usize) -> bool {
    levenshtein(query, term) <= max_distance
}

/// Filter a list of terms to those within `max_distance` of `query`.
pub fn fuzzy_filter<'a>(query: &str, terms: &[&'a str], max_distance: usize) -> Vec<&'a str> {
    terms.iter().filter(|&&t| is_fuzzy_match(query, t, max_distance)).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("hello", "hello"), 0);
        assert_eq!(levenshtein("hello", "helo"), 1);
    }

    #[test]
    fn fuzzy_filter_works() {
        let terms = vec!["hello", "hallo", "help", "world"];
        let matches = fuzzy_filter("hello", &terms, 1);
        assert!(matches.contains(&"hello"));
        assert!(matches.contains(&"hallo"));
        assert!(!matches.contains(&"world"));
    }
}
