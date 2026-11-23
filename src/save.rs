//! Save a copy of the model with edits applied.
//!
//! Untouched tensors stream their original bytes; only edited tensors are
//! re-encoded. Never writes into the source directory.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::edit::History;
use crate::model::safetensors::{verify_file, write_file_with_progress, TensorSource};
use crate::model::{ModelIndex, TensorBuf};

#[derive(Debug)]
pub struct SaveReport {
    pub dest: PathBuf,
    pub tensors: usize,
    pub edits: usize,
}

pub fn default_save_dir(index: &ModelIndex) -> PathBuf {
    let spec = &index.source_spec;
    let slug = if index.root.is_file() {
        index
            .root
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string())
    } else {
        Path::new(spec)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| spec.split(':').next().unwrap_or("model").replace('/', "--"))
    };
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(format!("{slug}-lobotomized"))
}

pub fn save_copy(
    index: &ModelIndex,
    edited: &std::collections::HashMap<String, Arc<TensorBuf>>,
    history: &History,
    out: &Path,
) -> Result<SaveReport> {
    let dirty = history.dirty_tensors();
    save_copy_inner(index, edited, &dirty, history.log(), out, |_, _, _| {})
}

pub fn save_copy_with_progress(
    index: &ModelIndex,
    edited: &std::collections::HashMap<String, Arc<TensorBuf>>,
    dirty: &HashSet<String>,
    log: &[Value],
    out: &Path,
    progress: impl FnMut(u64, u64, &str),
) -> Result<SaveReport> {
    save_copy_inner(index, edited, dirty, log, out, progress)
}

fn save_copy_inner(
    index: &ModelIndex,
    edited: &std::collections::HashMap<String, Arc<TensorBuf>>,
    dirty: &HashSet<String>,
    log: &[Value],
    out: &Path,
    mut progress: impl FnMut(u64, u64, &str),
) -> Result<SaveReport> {
    if log.is_empty() {
        bail!("no edits yet — nothing to save");
    }
    let out_abs = absolute(out);
    let src_abs = absolute(&index.root);
    if out_abs == src_abs || out_abs.starts_with(&src_abs) {
        bail!(
            "refusing to save inside the source ({}); pick another directory",
            src_abs.display()
        );
    }
    let dest = out_abs.join("model.safetensors");
    if dest.exists() {
        bail!(
            "output already contains {}; choose a new directory to avoid overwriting it",
            dest.display()
        );
    }
    let diff_path = out_abs.join("lobotomy-diff.jsonl");
    if diff_path.exists() {
        bail!(
            "output already contains {}; choose a new directory to avoid overwriting it",
            diff_path.display()
        );
    }
    fs::create_dir_all(&out_abs).with_context(|| format!("create {}", out_abs.display()))?;
    let model_temp = temporary_path(&out_abs, "model.safetensors");
    let diff_temp = temporary_path(&out_abs, "lobotomy-diff.jsonl");
    if model_temp.exists() || diff_temp.exists() {
        bail!(
            "temporary save path already exists in {}; retry with another output directory",
            out_abs.display()
        );
    }

    // Build sources in name order (index.tensors is already sorted).
    let mut sources: Vec<(String, String, Vec<usize>, TensorSource)> = Vec::new();
    for meta in &index.tensors {
        let dtype = meta.dtype.as_str().to_string();
        if dirty.contains(&meta.name) {
            let buf = edited
                .get(&meta.name)
                .with_context(|| format!("edited tensor {} is not in memory", meta.name))?;
            if !buf.is_editable() {
                bail!(
                    "tensor {} has non-float dtype {}",
                    meta.name,
                    meta.dtype.as_str()
                );
            }
            let bytes = meta.dtype.encode_f32(&buf.data);
            sources.push((
                meta.name.clone(),
                dtype,
                meta.shape.clone(),
                TensorSource::Bytes(bytes),
            ));
        } else {
            sources.push((
                meta.name.clone(),
                dtype,
                meta.shape.clone(),
                TensorSource::Raw {
                    shard: meta.shard.clone(),
                    start: meta.data_start,
                    len: meta.data_len,
                },
            ));
        }
    }

    let result = (|| {
        write_file_with_progress(&model_temp, &sources, |done, total, name| {
            progress(done, total, name)
        })
        .with_context(|| format!("write {}", model_temp.display()))?;
        let total_bytes = sources.iter().fold(0u64, |total, (_, _, _, source)| {
            total
                + match source {
                    TensorSource::Raw { len, .. } => *len,
                    TensorSource::Bytes(bytes) => bytes.len() as u64,
                }
        });
        progress(total_bytes, total_bytes, "verifying output");
        let expected: Vec<String> = index.tensors.iter().map(|t| t.name.clone()).collect();
        verify_file(&model_temp, &expected)?;

        copy_sidecars(&index.root, &out_abs)?;

        {
            let mut f = fs::File::create(&diff_temp)
                .with_context(|| format!("create {}", diff_temp.display()))?;
            for line in log {
                writeln!(f, "{}", serde_json::to_string(line)?)?;
            }
        }
        fs::rename(&diff_temp, &diff_path)
            .with_context(|| format!("finalize {}", diff_path.display()))?;
        if let Err(error) = fs::rename(&model_temp, &dest) {
            let _ = fs::remove_file(&diff_path);
            return Err(error).with_context(|| format!("finalize {}", dest.display()));
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&model_temp);
        let _ = fs::remove_file(&diff_temp);
    }
    result?;

    Ok(SaveReport {
        dest: out_abs,
        tensors: sources.len(),
        edits: log.len(),
    })
}

fn temporary_path(directory: &Path, stem: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    directory.join(format!(".{stem}.{}.{}.tmp", std::process::id(), nanos))
}

/// Resolve symlinks where possible so `/var` vs `/private/var` cannot
/// defeat the containment check.
fn absolute(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    if let Some(parent) = p.parent() {
        if let Ok(c) = std::fs::canonicalize(parent) {
            if let Some(name) = p.file_name() {
                return c.join(name);
            }
        }
    }
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn copy_sidecars(root: &Path, out: &Path) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let keep = name == "config.json"
            || name == "special_tokens_map.json"
            || name == "generation_config.json"
            || name.starts_with("tokenizer");
        if keep {
            let dest = out.join(name.as_ref());
            if !dest.exists() {
                fs::copy(entry.path(), &dest)?;
            }
        }
    }
    Ok(())
}
