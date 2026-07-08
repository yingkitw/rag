use crate::errors::{RagError, Result};
use crate::id::new_uuid_v4;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub name: String,
    pub properties: HashMap<String, String>,
}

impl GraphNode {
    pub fn new(name: String, label: String) -> Self {
        Self {
            id: new_uuid_v4(),
            label,
            name,
            properties: HashMap::new(),
        }
    }

    pub fn with_id(mut self, id: String) -> Self {
        self.id = id;
        self
    }

    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f32,
    pub properties: HashMap<String, String>,
}

impl GraphEdge {
    pub fn new(source: String, target: String, relation: String) -> Self {
        Self {
            id: new_uuid_v4(),
            source,
            target,
            relation,
            weight: 1.0,
            properties: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
    pub total_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: usize,
    pub node_ids: Vec<String>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPersisted {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

pub struct GraphStore {
    nodes: RwLock<HashMap<String, GraphNode>>,
    edges: RwLock<HashMap<String, GraphEdge>>,
    out_edges: RwLock<HashMap<String, HashSet<String>>>,
    in_edges: RwLock<HashMap<String, HashSet<String>>>,
    name_index: RwLock<HashMap<String, String>>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            edges: RwLock::new(HashMap::new()),
            out_edges: RwLock::new(HashMap::new()),
            in_edges: RwLock::new(HashMap::new()),
            name_index: RwLock::new(HashMap::new()),
        }
    }

    pub fn add_node(&self, node: GraphNode) -> Result<()> {
        let id = node.id.clone();
        let name = node.name.to_lowercase();
        self.name_index.write().unwrap().insert(name, id.clone());
        self.nodes.write().unwrap().insert(id, node);
        Ok(())
    }

    pub fn get_node(&self, id: &str) -> Option<GraphNode> {
        self.nodes.read().unwrap().get(id).cloned()
    }

    pub fn get_node_by_name(&self, name: &str) -> Option<GraphNode> {
        let key = name.to_lowercase();
        let id = self.name_index.read().unwrap().get(&key).cloned();
        id.and_then(|id| self.get_node(&id))
    }

    pub fn update_node(&self, id: &str, node: GraphNode) -> Result<bool> {
        let mut nodes = self.nodes.write().unwrap();
        if nodes.contains_key(id) {
            nodes.insert(id.to_string(), node);
            Ok(true)
        } else {
            Err(RagError::GraphError(format!("Node not found: {}", id)))
        }
    }

    pub fn remove_node(&self, id: &str) -> Result<bool> {
        // Snapshot affected edge ids under read locks before mutating.
        let edge_ids: Vec<String> = {
            let mut ids = Vec::new();
            if let Some(s) = self.out_edges.read().unwrap().get(id) {
                ids.extend(s.iter().cloned());
            }
            if let Some(s) = self.in_edges.read().unwrap().get(id) {
                ids.extend(s.iter().cloned());
            }
            ids
        };

        let node = self.nodes.write().unwrap().remove(id);
        let Some(node) = node else {
            return Ok(false);
        };

        let name = node.name.to_lowercase();
        self.name_index.write().unwrap().remove(&name);
        self.out_edges.write().unwrap().remove(id);
        self.in_edges.write().unwrap().remove(id);

        // Removing each edge also cleans the *other* endpoint's adjacency set.
        for eid in &edge_ids {
            self.remove_edge_direct(eid);
        }

        Ok(true)
    }

    pub fn add_edge(&self, edge: GraphEdge) -> Result<()> {
        let source = edge.source.clone();
        let target = edge.target.clone();
        let id = edge.id.clone();

        {
            let nodes = self.nodes.read().unwrap();
            if !nodes.contains_key(&source) {
                return Err(RagError::GraphError(format!(
                    "Source node not found: {}",
                    source
                )));
            }
            if !nodes.contains_key(&target) {
                return Err(RagError::GraphError(format!(
                    "Target node not found: {}",
                    target
                )));
            }
        }

        self.edges.write().unwrap().insert(id.clone(), edge);

        self.out_edges
            .write()
            .unwrap()
            .entry(source)
            .or_default()
            .insert(id.clone());

        self.in_edges
            .write()
            .unwrap()
            .entry(target)
            .or_default()
            .insert(id);

        Ok(())
    }

    pub fn get_edge(&self, id: &str) -> Option<GraphEdge> {
        self.edges.read().unwrap().get(id).cloned()
    }

    pub fn remove_edge(&self, id: &str) -> bool {
        self.remove_edge_direct(id)
    }

    fn remove_edge_direct(&self, id: &str) -> bool {
        let edge = self.edges.write().unwrap().remove(id);
        let Some(edge) = edge else {
            return false;
        };
        if let Some(set) = self.out_edges.write().unwrap().get_mut(&edge.source) {
            set.remove(id);
        }
        if let Some(set) = self.in_edges.write().unwrap().get_mut(&edge.target) {
            set.remove(id);
        }
        true
    }

    pub fn upsert_edge(&self, edge: GraphEdge) -> Result<()> {
        let source = edge.source.clone();
        let target = edge.target.clone();
        let relation = edge.relation.clone();

        let existing = self.find_edge(&source, &target, &relation);
        if let Some(existing) = existing {
            self.remove_edge(&existing.id);
        }

        self.add_edge(edge)
    }

    pub fn find_edge(&self, source: &str, target: &str, relation: &str) -> Option<GraphEdge> {
        let edges = self.edges.read().unwrap();
        edges
            .values()
            .find(|e| e.source == source && e.target == target && e.relation == relation)
            .cloned()
    }

    pub fn neighbors(&self, node_id: &str) -> Vec<GraphNode> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        // out-edge targets first, then in-edge sources (deduplicated together).
        let candidate_ids: Vec<String> = {
            let mut ids = Vec::new();
            let out = self.out_edges.read().unwrap();
            if let Some(edge_ids) = out.get(node_id) {
                let edges = self.edges.read().unwrap();
                for eid in edge_ids.iter() {
                    if let Some(edge) = edges.get(eid) {
                        ids.push(edge.target.clone());
                    }
                }
            }
            drop(out);
            let ine = self.in_edges.read().unwrap();
            if let Some(edge_ids) = ine.get(node_id) {
                let edges = self.edges.read().unwrap();
                for eid in edge_ids.iter() {
                    if let Some(edge) = edges.get(eid) {
                        ids.push(edge.source.clone());
                    }
                }
            }
            ids
        };

        for nid in candidate_ids {
            if seen.insert(nid.clone()) {
                if let Some(node) = self.nodes.read().unwrap().get(&nid).cloned() {
                    result.push(node);
                }
            }
        }

        result
    }

    pub fn out_neighbors(&self, node_id: &str) -> Vec<GraphNode> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        let targets: Vec<String> = {
            let mut ids = Vec::new();
            let out = self.out_edges.read().unwrap();
            if let Some(edge_ids) = out.get(node_id) {
                let edges = self.edges.read().unwrap();
                for eid in edge_ids.iter() {
                    if let Some(edge) = edges.get(eid) {
                        ids.push(edge.target.clone());
                    }
                }
            }
            ids
        };

        for nid in targets {
            if seen.insert(nid.clone()) {
                if let Some(node) = self.nodes.read().unwrap().get(&nid).cloned() {
                    result.push(node);
                }
            }
        }

        result
    }

    pub fn in_neighbors(&self, node_id: &str) -> Vec<GraphNode> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        let sources: Vec<String> = {
            let mut ids = Vec::new();
            let ine = self.in_edges.read().unwrap();
            if let Some(edge_ids) = ine.get(node_id) {
                let edges = self.edges.read().unwrap();
                for eid in edge_ids.iter() {
                    if let Some(edge) = edges.get(eid) {
                        ids.push(edge.source.clone());
                    }
                }
            }
            ids
        };

        for nid in sources {
            if seen.insert(nid.clone()) {
                if let Some(node) = self.nodes.read().unwrap().get(&nid).cloned() {
                    result.push(node);
                }
            }
        }

        result
    }

    pub fn degree(&self, node_id: &str) -> usize {
        let out = self
            .out_edges
            .read()
            .unwrap()
            .get(node_id)
            .map(|s| s.len())
            .unwrap_or(0);
        let in_deg = self
            .in_edges
            .read()
            .unwrap()
            .get(node_id)
            .map(|s| s.len())
            .unwrap_or(0);
        out + in_deg
    }

    pub fn edges_between(&self, source: &str, target: &str) -> Vec<GraphEdge> {
        let mut result = Vec::new();

        let out = self.out_edges.read().unwrap();
        if let Some(edge_ids) = out.get(source) {
            let edges = self.edges.read().unwrap();
            for eid in edge_ids.iter() {
                if let Some(edge) = edges.get(eid)
                    && edge.target == target
                {
                    result.push(edge.clone());
                }
            }
        }

        result
    }

    pub fn nodes_by_label(&self, label: &str) -> Vec<GraphNode> {
        self.nodes
            .read()
            .unwrap()
            .values()
            .filter(|n| n.label == label)
            .cloned()
            .collect()
    }

    pub fn edges_by_relation(&self, relation: &str) -> Vec<GraphEdge> {
        self.edges
            .read()
            .unwrap()
            .values()
            .filter(|e| e.relation == relation)
            .cloned()
            .collect()
    }

    pub fn bfs(&self, start_id: &str, max_depth: usize) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(start_id.to_string());
        queue.push_back((start_id.to_string(), 0usize));

        while let Some((node_id, depth)) = queue.pop_front() {
            if depth > 0
                && let Some(node) = self.nodes.read().unwrap().get(&node_id).cloned()
            {
                result.push(node);
            }

            if depth < max_depth {
                for neighbor in self.neighbors(&node_id) {
                    if visited.insert(neighbor.id.clone()) {
                        queue.push_back((neighbor.id.clone(), depth + 1));
                    }
                }
            }
        }

        result
    }

    pub fn k_hop(&self, start_id: &str, k: usize) -> Vec<Vec<GraphNode>> {
        let mut levels = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_id.to_string());
        let mut current_level = vec![start_id.to_string()];

        for _ in 0..k {
            let mut next_level_nodes = Vec::new();
            let mut next_level_ids = Vec::new();

            for nid in &current_level {
                for neighbor in self.neighbors(nid) {
                    if visited.insert(neighbor.id.clone()) {
                        next_level_nodes.push(neighbor.clone());
                        next_level_ids.push(neighbor.id.clone());
                    }
                }
            }

            levels.push(next_level_nodes);
            current_level = next_level_ids;

            if current_level.is_empty() {
                break;
            }
        }

        levels
    }

    pub fn shortest_path(&self, source: &str, target: &str) -> Option<GraphPath> {
        if source == target {
            return Some(GraphPath {
                node_ids: vec![source.to_string()],
                edge_ids: vec![],
                total_weight: 0.0,
            });
        }

        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(source.to_string());
        visited.insert(source.to_string(), (None::<String>, None::<String>, 0.0f32));

        while let Some(current) = queue.pop_front() {
            if current == target {
                let mut path_nodes = Vec::new();
                let mut path_edges = Vec::new();
                let mut total_weight = 0.0f32;
                let mut node = Some(target.to_string());

                while let Some(n) = node {
                    if let Some((prev_node, edge_id, weight)) = visited.get(&n) {
                        if let Some(eid) = edge_id {
                            path_edges.push(eid.clone());
                        }
                        total_weight += weight;
                        path_nodes.push(n.clone());
                        node = prev_node.clone();
                    } else {
                        path_nodes.push(n.clone());
                        break;
                    }
                }

                path_nodes.reverse();
                path_edges.reverse();

                return Some(GraphPath {
                    node_ids: path_nodes,
                    edge_ids: path_edges,
                    total_weight,
                });
            }

            {
                let out = self.out_edges.read().unwrap();
                if let Some(edge_ids) = out.get(&current).cloned() {
                    let edges = self.edges.read().unwrap();
                    for eid in edge_ids.iter() {
                        if let Some(edge) = edges.get(eid)
                            && !visited.contains_key(&edge.target)
                        {
                            visited.insert(
                                edge.target.clone(),
                                (Some(current.clone()), Some(eid.clone()), edge.weight),
                            );
                            queue.push_back(edge.target.clone());
                        }
                    }
                }
            }

            {
                let ine = self.in_edges.read().unwrap();
                if let Some(edge_ids) = ine.get(&current).cloned() {
                    let edges = self.edges.read().unwrap();
                    for eid in edge_ids.iter() {
                        if let Some(edge) = edges.get(eid)
                            && !visited.contains_key(&edge.source)
                        {
                            visited.insert(
                                edge.source.clone(),
                                (Some(current.clone()), Some(eid.clone()), edge.weight),
                            );
                            queue.push_back(edge.source.clone());
                        }
                    }
                }
            }
        }

        None
    }

    pub fn detect_communities(&self) -> Vec<Community> {
        let node_ids: Vec<String> = self.nodes.read().unwrap().keys().cloned().collect();

        if node_ids.is_empty() {
            return Vec::new();
        }

        let mut labels: HashMap<String, usize> = HashMap::new();
        for (i, id) in node_ids.iter().enumerate() {
            labels.insert(id.clone(), i);
        }

        let max_iterations = 20;
        for _ in 0..max_iterations {
            let mut changed = false;

            for node_id in &node_ids {
                let neighbor_ids: Vec<String> =
                    self.neighbors(node_id).into_iter().map(|n| n.id).collect();

                if neighbor_ids.is_empty() {
                    continue;
                }

                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for nid in &neighbor_ids {
                    if let Some(label) = labels.get(nid) {
                        *label_counts.entry(*label).or_insert(0) += 1;
                    }
                }

                if let Some(best_label) = label_counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(label, _)| label)
                    && labels.get(node_id) != Some(&best_label)
                {
                    labels.insert(node_id.clone(), best_label);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut community_map: HashMap<usize, Vec<String>> = HashMap::new();
        for (node_id, label) in &labels {
            community_map
                .entry(*label)
                .or_default()
                .push(node_id.clone());
        }

        community_map
            .into_iter()
            .map(|(id, node_ids)| Community {
                id,
                size: node_ids.len(),
                node_ids,
            })
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.read().unwrap().len()
    }

    pub fn density(&self) -> f64 {
        let n = self.nodes.read().unwrap().len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let max_edges = n * (n - 1.0);
        self.edges.read().unwrap().len() as f64 / max_edges
    }

    pub fn all_nodes(&self) -> Vec<GraphNode> {
        self.nodes.read().unwrap().values().cloned().collect()
    }

    pub fn all_edges(&self) -> Vec<GraphEdge> {
        self.edges.read().unwrap().values().cloned().collect()
    }

    pub fn clear(&self) {
        self.nodes.write().unwrap().clear();
        self.edges.write().unwrap().clear();
        self.out_edges.write().unwrap().clear();
        self.in_edges.write().unwrap().clear();
        self.name_index.write().unwrap().clear();
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let data = GraphPersisted {
            nodes: self.all_nodes(),
            edges: self.all_edges(),
        };
        let json = serde_json::to_string_pretty(&data)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let data: GraphPersisted = serde_json::from_str(&content)?;
        Self::from_persisted(data)
    }

    /// Restore a graph from a [`GraphPersisted`] payload (for example after JSON decode).
    pub fn from_persisted(data: GraphPersisted) -> Result<Self> {
        let store = Self::new();

        for node in data.nodes {
            let id = node.id.clone();
            let name = node.name.to_lowercase();
            store.name_index.write().unwrap().insert(name, id.clone());
            store.nodes.write().unwrap().insert(id, node);
        }

        for edge in data.edges {
            let id = edge.id.clone();
            let source = edge.source.clone();
            let target = edge.target.clone();
            store
                .out_edges
                .write()
                .unwrap()
                .entry(source)
                .or_default()
                .insert(id.clone());
            store
                .in_edges
                .write()
                .unwrap()
                .entry(target)
                .or_default()
                .insert(id.clone());
            store.edges.write().unwrap().insert(id, edge);
        }

        Ok(store)
    }

    pub fn subgraph(&self, node_ids: &[String]) -> Self {
        let sub = Self::new();
        let node_set: HashSet<&String> = node_ids.iter().collect();

        for nid in node_ids {
            if let Some(node) = self.get_node(nid) {
                let _ = sub.add_node(node);
            }
        }

        for edge in self.all_edges() {
            if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
                let _ = sub.add_edge(edge);
            }
        }

        sub
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_node() {
        let store = GraphStore::new();
        let node = GraphNode::new("Alice".to_string(), "person".to_string());
        let id = node.id.clone();
        store.add_node(node).unwrap();

        let retrieved = store.get_node(&id).unwrap();
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.label, "person");
    }

    #[test]
    fn test_get_node_by_name() {
        let store = GraphStore::new();
        let node = GraphNode::new("New York".to_string(), "location".to_string());
        store.add_node(node).unwrap();

        let retrieved = store.get_node_by_name("New York").unwrap();
        assert_eq!(retrieved.label, "location");

        let retrieved_lower = store.get_node_by_name("new york").unwrap();
        assert_eq!(retrieved_lower.name, "New York");
    }

    #[test]
    fn test_remove_node() {
        let store = GraphStore::new();
        let node = GraphNode::new("Alice".to_string(), "person".to_string());
        let id = node.id.clone();
        store.add_node(node).unwrap();

        assert!(store.remove_node(&id).unwrap());
        assert!(store.get_node(&id).is_none());
        assert!(store.get_node_by_name("Alice").is_none());
    }

    #[test]
    fn test_add_and_get_edge() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "entity".to_string());
        let b = GraphNode::new("B".to_string(), "entity".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        let edge = GraphEdge::new(a_id.clone(), b_id.clone(), "connects".to_string());
        let e_id = edge.id.clone();
        store.add_edge(edge).unwrap();

        let retrieved = store.get_edge(&e_id).unwrap();
        assert_eq!(retrieved.source, a_id);
        assert_eq!(retrieved.target, b_id);
        assert_eq!(retrieved.relation, "connects");
    }

    #[test]
    fn test_add_edge_missing_node() {
        let store = GraphStore::new();
        let edge = GraphEdge::new(
            "nonexistent".to_string(),
            "also".to_string(),
            "x".to_string(),
        );
        assert!(store.add_edge(edge).is_err());
    }

    #[test]
    fn test_remove_edge() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        let edge = GraphEdge::new(a_id, b_id, "rel".to_string());
        let e_id = edge.id.clone();
        store.add_edge(edge).unwrap();

        assert!(store.remove_edge(&e_id));
        assert!(store.get_edge(&e_id).is_none());
    }

    #[test]
    fn test_neighbors() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(
                a_id.clone(),
                b_id.clone(),
                "knows".to_string(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(
                c_id.clone(),
                a_id.clone(),
                "knows".to_string(),
            ))
            .unwrap();

        let neighbors = store.neighbors(&a_id);
        assert_eq!(neighbors.len(), 2);
        let names: Vec<&str> = neighbors.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"B"));
        assert!(names.contains(&"C"));
    }

    #[test]
    fn test_out_neighbors() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        store
            .add_edge(GraphEdge::new(
                a_id.clone(),
                b_id.clone(),
                "follows".to_string(),
            ))
            .unwrap();

        let out = store.out_neighbors(&a_id);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "B");

        let out_b = store.out_neighbors(&b_id);
        assert!(out_b.is_empty());
    }

    #[test]
    fn test_degree() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id, "knows".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(c_id, a_id.clone(), "knows".to_string()))
            .unwrap();

        assert_eq!(store.degree(&a_id), 2);
    }

    #[test]
    fn test_bfs() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let d = GraphNode::new("D".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id.clone(), c_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(a_id.clone(), d_id, "e".to_string()))
            .unwrap();

        let reachable = store.bfs(&a_id, 2);
        assert_eq!(reachable.len(), 3);
        let names: Vec<&str> = reachable.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"B"));
        assert!(names.contains(&"C"));
        assert!(names.contains(&"D"));
    }

    #[test]
    fn test_k_hop() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id, c_id, "e".to_string()))
            .unwrap();

        let levels = store.k_hop(&a_id, 2);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[0][0].name, "B");
        assert_eq!(levels[1].len(), 1);
        assert_eq!(levels[1][0].name, "C");
    }

    #[test]
    fn test_shortest_path() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id.clone(), c_id.clone(), "e".to_string()))
            .unwrap();

        let path = store.shortest_path(&a_id, &c_id).unwrap();
        assert_eq!(path.node_ids.len(), 3);
        assert_eq!(path.node_ids[0], a_id);
        assert_eq!(path.node_ids[2], c_id);
    }

    #[test]
    fn test_shortest_path_not_found() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        assert!(store.shortest_path(&a_id, &b_id).is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let a_id = a.id.clone();
        store.add_node(a).unwrap();

        let path = store.shortest_path(&a_id, &a_id).unwrap();
        assert_eq!(path.node_ids.len(), 1);
        assert_eq!(path.total_weight, 0.0);
    }

    #[test]
    fn test_detect_communities() {
        let store = GraphStore::new();

        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let d = GraphNode::new("D".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        let d_id = d.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();
        store.add_node(d).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id.clone(), a_id, "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(c_id.clone(), d_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(d_id, c_id, "e".to_string()))
            .unwrap();

        let communities = store.detect_communities();
        assert_eq!(communities.len(), 2);

        let sizes: Vec<usize> = communities.iter().map(|c| c.size).collect();
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn test_nodes_by_label() {
        let store = GraphStore::new();
        store
            .add_node(GraphNode::new("Alice".to_string(), "person".to_string()))
            .unwrap();
        store
            .add_node(GraphNode::new("Bob".to_string(), "person".to_string()))
            .unwrap();
        store
            .add_node(GraphNode::new("Paris".to_string(), "location".to_string()))
            .unwrap();

        let people = store.nodes_by_label("person");
        assert_eq!(people.len(), 2);

        let locations = store.nodes_by_label("location");
        assert_eq!(locations.len(), 1);
    }

    #[test]
    fn test_edges_by_relation() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        store
            .add_edge(GraphEdge::new(
                a_id.clone(),
                b_id.clone(),
                "friend".to_string(),
            ))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id, a_id, "colleague".to_string()))
            .unwrap();

        let friends = store.edges_by_relation("friend");
        assert_eq!(friends.len(), 1);
        let colleagues = store.edges_by_relation("colleague");
        assert_eq!(colleagues.len(), 1);
    }

    #[test]
    fn test_density() {
        let store = GraphStore::new();
        assert_eq!(store.density(), 0.0);

        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        store
            .add_edge(GraphEdge::new(a_id, b_id, "e".to_string()))
            .unwrap();

        let density = store.density();
        assert!(density > 0.0 && density <= 1.0);
    }

    #[test]
    fn test_clear() {
        let store = GraphStore::new();
        store
            .add_node(GraphNode::new("A".to_string(), "e".to_string()))
            .unwrap();
        store.clear();
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_subgraph() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let c = GraphNode::new("C".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        let c_id = c.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store.add_node(c).unwrap();

        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id.clone(), "e".to_string()))
            .unwrap();
        store
            .add_edge(GraphEdge::new(b_id.clone(), c_id, "e".to_string()))
            .unwrap();

        let sub = store.subgraph(&[a_id.clone(), b_id.clone()]);
        assert_eq!(sub.node_count(), 2);
        assert_eq!(sub.edge_count(), 1);
    }

    #[test]
    fn test_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.json");

        let store = GraphStore::new();
        let a = GraphNode::new("Alice".to_string(), "person".to_string());
        let b = GraphNode::new("Bob".to_string(), "person".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();
        store
            .add_edge(GraphEdge::new(a_id.clone(), b_id, "knows".to_string()))
            .unwrap();

        store.save_to_file(&path).unwrap();
        let loaded = GraphStore::load_from_file(&path).unwrap();

        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.edge_count(), 1);
        assert!(loaded.get_node_by_name("Alice").is_some());
        assert!(loaded.get_node_by_name("Bob").is_some());
    }

    #[test]
    fn test_upsert_edge() {
        let store = GraphStore::new();
        let a = GraphNode::new("A".to_string(), "e".to_string());
        let b = GraphNode::new("B".to_string(), "e".to_string());
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        store.add_node(a).unwrap();
        store.add_node(b).unwrap();

        let edge1 = GraphEdge::new(a_id.clone(), b_id.clone(), "rel".to_string()).with_weight(1.0);
        store.upsert_edge(edge1).unwrap();
        assert_eq!(store.edge_count(), 1);

        let edge2 = GraphEdge::new(a_id.clone(), b_id.clone(), "rel".to_string()).with_weight(2.0);
        store.upsert_edge(edge2).unwrap();
        assert_eq!(store.edge_count(), 1);

        let edges = store.edges_between(&a_id, &b_id);
        assert_eq!(edges.len(), 1);
        assert!((edges[0].weight - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_node_with_property() {
        let node = GraphNode::new("test".to_string(), "type".to_string())
            .with_property("key".to_string(), "value".to_string());
        assert_eq!(node.properties.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_edge_with_weight() {
        let edge =
            GraphEdge::new("a".to_string(), "b".to_string(), "rel".to_string()).with_weight(3.5);
        assert!((edge.weight - 3.5).abs() < 0.01);
    }
}
