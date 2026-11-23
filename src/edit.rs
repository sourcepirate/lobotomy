//! Tensor edit operations and undo/redo history.

use std::collections::HashSet;

use crate::model::stats::compute_stats;
use crate::model::TensorBuf;

#[derive(Debug, Clone)]
pub enum EditOp {
    Scale { factor: f32 },
    Noise { sigma: f32, seed: u64 },
    Zero,
    Prune { threshold: f32 },
    Clamp { lo: f32, hi: f32 },
    SetCell { index: usize, value: f32 },
}

impl EditOp {
    pub fn label(&self) -> String {
        match self {
            EditOp::Scale { factor } => format!("scale {factor}"),
            EditOp::Noise { sigma, .. } => format!("noise {sigma}"),
            EditOp::Zero => "zero".to_string(),
            EditOp::Prune { threshold } => format!("prune {threshold}"),
            EditOp::Clamp { lo, hi } => format!("clamp {lo} {hi}"),
            EditOp::SetCell { index, value } => format!("set [{index}] {value}"),
        }
    }

    /// Ops that destroy information need a snapshot for undo.
    pub fn needs_snapshot(&self) -> bool {
        match self {
            EditOp::Zero | EditOp::Prune { .. } | EditOp::Clamp { .. } | EditOp::Noise { .. } => {
                true
            }
            EditOp::Scale { factor } => *factor == 0.0,
            EditOp::SetCell { .. } => false,
        }
    }

    pub fn apply(&self, data: &mut [f32]) {
        match self {
            EditOp::Scale { factor } => {
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
            EditOp::Noise { sigma, seed } => {
                // deterministic xorshift so redo reproduces the exact noise
                let mut state = (*seed).max(1);
                for v in data.iter_mut() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 40) as f32 / 16_777_216.0; // [0,1)
                    let g = (u - 0.5) * 6.0; // cheap approximate gaussian
                    *v += g * sigma;
                }
            }
            EditOp::Zero => {
                for v in data.iter_mut() {
                    *v = 0.0;
                }
            }
            EditOp::Prune { threshold } => {
                for v in data.iter_mut() {
                    if v.abs() < *threshold {
                        *v = 0.0;
                    }
                }
            }
            EditOp::Clamp { lo, hi } => {
                for v in data.iter_mut() {
                    *v = v.clamp(*lo, *hi);
                }
            }
            EditOp::SetCell { index, value } => {
                if let Some(slot) = data.get_mut(*index) {
                    *slot = *value;
                }
            }
        }
    }

    /// Inverse for ops that can be reversed without a snapshot.
    pub fn inverse(&self, data: &mut [f32], old: Option<f32>) {
        match self {
            EditOp::Scale { factor } if *factor != 0.0 => {
                let inv = 1.0 / factor;
                for v in data.iter_mut() {
                    *v *= inv;
                }
            }
            EditOp::SetCell { index, .. } => {
                if let (Some(slot), Some(prev)) = (data.get_mut(*index), old) {
                    *slot = prev;
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edit {
    pub tensor: String,
    pub op: EditOp,
    pub prev: Option<Vec<f32>>,
    /// Previous value for a single-cell edit.
    pub old: Option<f32>,
    pub diff: serde_json::Value,
}

pub struct History {
    undo: Vec<Edit>,
    redo: Vec<Edit>,
    log: Vec<serde_json::Value>,
    dirty: HashSet<String>,
    noised: u64,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    pub fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            log: Vec::new(),
            dirty: HashSet::new(),
            noised: 0,
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    pub fn log(&self) -> &[serde_json::Value] {
        &self.log
    }

    pub fn is_dirty(&self, tensor: &str) -> bool {
        self.dirty.contains(tensor)
    }

    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    pub fn dirty_tensors(&self) -> std::collections::HashSet<String> {
        self.dirty.clone()
    }

    pub fn has_edits(&self) -> bool {
        !self.log.is_empty()
    }

    pub fn next_seed(&mut self) -> u64 {
        self.noised += 1;
        (self.noised * 0x9E37_79B9_7F4A_7C15) | 1
    }

    pub fn recent(&self, n: usize) -> Vec<String> {
        self.undo
            .iter()
            .rev()
            .take(n)
            .map(|e| format!("{} · {}", e.op.label(), short(&e.tensor)))
            .collect()
    }

    /// Apply an op, recording enough state to undo and redo it.
    pub fn apply(&mut self, buf: &mut TensorBuf, op: EditOp) -> String {
        if !buf.is_editable() {
            return format!("{} is {} (read-only)", short(&buf.name), buf.dtype.as_str());
        }
        let before = compute_stats(buf);
        let old = match &op {
            EditOp::SetCell { index, .. } => buf.data.get(*index).copied(),
            _ => None,
        };
        let prev = if op.needs_snapshot() {
            Some(buf.data.clone())
        } else {
            None
        };
        op.apply(&mut buf.data);
        let after = compute_stats(buf);
        let diff = serde_json::json!({
            "ts": now_iso8601(),
            "tensor": buf.name,
            "op": op.label(),
        });
        self.log.push(diff.clone());
        self.dirty.insert(buf.name.clone());
        self.undo.push(Edit {
            tensor: buf.name.clone(),
            op,
            prev,
            old,
            diff,
        });
        self.redo.clear();
        format!(
            "{} · mean {:.3e}→{:.3e} · sparsity {:.1}%→{:.1}%",
            short(&buf.name),
            before.mean,
            after.mean,
            before.sparsity * 100.0,
            after.sparsity * 100.0
        )
    }

    /// Undo the last edit, mutating `buf` in place. Returns a status message.
    /// Tensor that the next undo would affect.
    pub fn undo_tensor(&self) -> Option<&str> {
        self.undo.last().map(|e| e.tensor.as_str())
    }

    /// Tensor that the next redo would affect.
    pub fn redo_tensor(&self) -> Option<&str> {
        self.redo.last().map(|e| e.tensor.as_str())
    }

    pub fn undo(&mut self, buf: &mut TensorBuf) -> String {
        let Some(edit) = self.undo.pop() else {
            return "nothing to undo".to_string();
        };
        if let Some(prev) = &edit.prev {
            buf.data.clone_from(prev);
        } else {
            edit.op.inverse(&mut buf.data, edit.old);
        }
        let msg = format!("undo {}", edit.op.label());
        self.redo.push(edit);
        msg
    }

    pub fn redo(&mut self, buf: &mut TensorBuf) -> String {
        let Some(edit) = self.redo.pop() else {
            return "nothing to redo".to_string();
        };
        edit.op.apply(&mut buf.data);
        let msg = format!("redo {}", edit.op.label());
        self.undo.push(edit);
        msg
    }
}

pub fn short(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() <= 2 {
        name.to_string()
    } else {
        parts[parts.len() - 2..].join(".")
    }
}

fn now_iso8601() -> String {
    // seconds since epoch is enough for a diff log; avoids a chrono dependency
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("{}", d.as_secs()),
        Err(_) => "0".to_string(),
    }
}
