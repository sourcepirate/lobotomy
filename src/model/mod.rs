//! Model index, natural ordering, tensor loading and a small LRU cache.

pub mod resolve;
pub mod safetensors;
pub mod stats;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;

use self::safetensors::{DType, Shard, TensorMeta};

pub use self::safetensors::TensorMeta as Meta;

#[derive(Debug, Clone)]
pub struct ModelIndex {
    pub root: PathBuf,
    pub shards: Vec<PathBuf>,
    pub tensors: Vec<TensorMeta>,
    pub by_name: HashMap<String, usize>,
    pub config: Value,
    pub arch: String,
    pub source_spec: String,
}

impl ModelIndex {
    pub fn names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|i| &self.tensors[*i])
    }
}

/// Natural comparison so `layers.3 < layers.4 < layers.23`.
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    fn tokenize(s: &str) -> Vec<(bool, u128, String)> {
        // (is_number, numeric value, lowercase text)
        let mut out = Vec::new();
        let mut text = String::new();
        let mut digits = String::new();
        for ch in s.chars() {
            if ch.is_ascii_digit() {
                if !text.is_empty() {
                    out.push((false, 0, text.to_lowercase()));
                    text.clear();
                }
                digits.push(ch);
            } else {
                if !digits.is_empty() {
                    out.push((true, digits.parse().unwrap_or(0), String::new()));
                    digits.clear();
                }
                text.push(ch);
            }
        }
        if !digits.is_empty() {
            out.push((true, digits.parse().unwrap_or(0), String::new()));
        }
        if !text.is_empty() {
            out.push((false, 0, text.to_lowercase()));
        }
        out
    }

    let ta = tokenize(a);
    let tb = tokenize(b);
    for (x, y) in ta.iter().zip(tb.iter()) {
        let ord = match (x.0, y.0) {
            (true, true) => x.1.cmp(&y.1),
            (false, false) => x.2.cmp(&y.2),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
        };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    ta.len().cmp(&tb.len())
}

/// Group tensors into layer nodes, ordered naturally.
pub fn group_by_layer(names: &[String]) -> Vec<(String, Vec<String>)> {
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for name in names {
        let key = group_key(name);
        match index.get(&key) {
            Some(i) => groups[*i].1.push(name.clone()),
            None => {
                index.insert(key.clone(), groups.len());
                groups.push((key, vec![name.clone()]));
            }
        }
    }
    for (_, members) in groups.iter_mut() {
        members.sort_by(|a, b| natural_cmp(a, b));
    }
    groups.sort_by(|a, b| natural_cmp(&a.0, &b.0));
    groups
}

fn group_key(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len().saturating_sub(1) {
        let is_layer_word = matches!(parts[i], "layers" | "layer" | "h" | "blocks" | "block");
        if is_layer_word
            && parts[i + 1].chars().all(|c| c.is_ascii_digit())
            && !parts[i + 1].is_empty()
        {
            return parts[..=i + 1].join(".");
        }
    }
    if parts.len() >= 3 {
        parts[..2].join(".")
    } else if parts.len() == 2 {
        parts[0].to_string()
    } else {
        "(root)".to_string()
    }
}

/// Leaf label relative to its group node (`self_attn.q_proj.weight`).
pub fn relative_label(group: &str, name: &str) -> String {
    let prefix = format!("{group}.");
    if let Some(rest) = name.strip_prefix(&prefix) {
        return rest.to_string();
    }
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 {
        parts[parts.len() - 2..].join(".")
    } else {
        name.to_string()
    }
}

pub fn build_index(spec: &str, revision: Option<&str>, offline: bool) -> Result<ModelIndex> {
    let root = resolve::resolve_source(spec, revision, offline)?;
    let mut shards: Vec<PathBuf> = if root.is_file() {
        vec![root.clone()]
    } else {
        std::fs::read_dir(&root)
            .with_context(|| format!("read model dir {}", root.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect()
    };
    shards.sort_by(|a, b| natural_cmp(&a.display().to_string(), &b.display().to_string()));
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files under {}", root.display());
    }
    let mut tensors = Vec::new();
    for shard in &shards {
        tensors.extend(safetensors::read_shard(shard)?);
    }
    tensors.sort_by(|a, b| natural_cmp(&a.name, &b.name));
    let mut by_name = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        by_name.insert(t.name.clone(), i);
    }
    let config = root
        .join("config.json")
        .exists()
        .then(|| std::fs::read_to_string(root.join("config.json")).ok())
        .flatten()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or(Value::Null);
    let arch = config["model_type"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    Ok(ModelIndex {
        root,
        shards,
        tensors,
        by_name,
        config,
        arch,
        source_spec: spec.to_string(),
    })
}

/// An editable tensor held in an f32 working buffer.
#[derive(Debug, Clone)]
pub struct TensorBuf {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<f32>,
}

impl TensorBuf {
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn is_editable(&self) -> bool {
        self.dtype.is_editable()
    }

    /// Value at a multi-dimensional index, with wrapping.
    pub fn get(&self, index: &[usize]) -> Option<f32> {
        let flat = self.flat_index(index)?;
        self.data.get(flat).copied()
    }

    pub fn flat_index(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (i, dim) in self.shape.iter().enumerate().rev() {
            let idx = *index.get(i)?;
            if idx >= *dim {
                return None;
            }
            flat += idx * stride;
            stride *= dim;
        }
        Some(flat)
    }

    /// Row-major strides for the last two dims (used by the heatmap).
    pub fn strides(&self) -> Vec<usize> {
        let mut out = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            out[i] = out[i + 1] * self.shape[i + 1];
        }
        out
    }
}

/// Loads tensors on demand, keeping a bounded number in memory.
pub struct TensorStore {
    shards: HashMap<PathBuf, Arc<Shard>>,
    cache: HashMap<String, Arc<TensorBuf>>,
    order: std::collections::VecDeque<String>,
    budget: usize,
    bytes: usize,
}

impl TensorStore {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            shards: HashMap::new(),
            cache: HashMap::new(),
            order: std::collections::VecDeque::new(),
            budget: budget_bytes,
            bytes: 0,
        }
    }

    pub fn get(&mut self, meta: &TensorMeta) -> Result<Arc<TensorBuf>> {
        if let Some(t) = self.cache.get(&meta.name) {
            return Ok(t.clone());
        }
        let shard = match self.shards.get(&meta.shard) {
            Some(s) => s.clone(),
            None => {
                let s = Arc::new(Shard::open(&meta.shard)?);
                self.shards.insert(meta.shard.clone(), s.clone());
                s
            }
        };
        let bytes = shard.bytes(meta)?;
        let data = meta.dtype.decode_f32(bytes);
        if data.is_empty() && meta.numel() > 0 {
            anyhow::bail!(
                "tensor {} has dtype {} which cannot be edited as floats",
                meta.name,
                meta.dtype.as_str()
            );
        }
        let buf = Arc::new(TensorBuf {
            name: meta.name.clone(),
            shape: meta.shape.clone(),
            dtype: meta.dtype.clone(),
            data,
        });
        let size = buf.data.len() * 4;
        self.cache.insert(meta.name.clone(), buf.clone());
        self.order.push_back(meta.name.clone());
        self.bytes += size;
        while self.bytes > self.budget && self.order.len() > 1 {
            if let Some(old) = self.order.pop_front() {
                if let Some(removed) = self.cache.remove(&old) {
                    self.bytes -= removed.data.len() * 4;
                }
            }
        }
        Ok(buf)
    }

    pub fn insert(&mut self, buf: TensorBuf) {
        let size = buf.data.len() * 4;
        let name = buf.name.clone();
        if self.cache.insert(name.clone(), Arc::new(buf)).is_none() {
            self.order.push_back(name);
            self.bytes += size;
        }
    }

    pub fn cached(&self, name: &str) -> Option<Arc<TensorBuf>> {
        self.cache.get(name).cloned()
    }

    pub fn cached_names(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
}
