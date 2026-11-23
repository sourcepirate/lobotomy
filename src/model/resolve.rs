//! Resolve a model spec to a local directory.
//!
//! Accepts a local path or an HF repo id (`org/model`, optionally
//! `:rev` or `@rev`). Repo ids are downloaded once into
//! `~/.cache/lobotomy/<org>--<model>/<rev>/`; everything after that is local.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

const HF_API: &str = "https://huggingface.co/api/models";
const HF_RESOLVE: &str = "https://huggingface.co";

pub fn is_repo_id(spec: &str) -> bool {
    if Path::new(spec).exists() {
        return false;
    }
    let base = split_revision(spec).0;
    let mut parts = base.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), None) => {
            !a.is_empty()
                && !b.is_empty()
                && a.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
                && b.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        }
        _ => false,
    }
}

/// Split `org/model[:rev]` / `org/model@rev`.
pub fn split_revision(spec: &str) -> (String, Option<String>) {
    for sep in ['@', ':'] {
        if let Some(idx) = spec.rfind(sep) {
            let (head, tail) = spec.split_at(idx);
            let rev = &tail[1..];
            if head.contains('/') && !rev.is_empty() {
                return (head.to_string(), Some(rev.to_string()));
            }
        }
    }
    (spec.to_string(), None)
}

pub fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("LOBOTOMY_CACHE") {
        return PathBuf::from(dir);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("lobotomy")
}

fn slug(repo: &str) -> String {
    repo.replace('/', "--")
}

/// Return a local directory holding `config.json` + `*.safetensors`.
pub fn resolve_source(spec: &str, revision: Option<&str>, offline: bool) -> Result<PathBuf> {
    let p = PathBuf::from(spec);
    if p.exists() {
        return p.canonicalize().context("canonicalize model path");
    }
    if !is_repo_id(spec) {
        bail!("model not found: {spec:?} (pass a path or a repo id like Qwen/Qwen2.5-0.5B)");
    }
    let (repo, rev_from_spec) = split_revision(spec);
    let rev = revision
        .map(str::to_string)
        .or(rev_from_spec)
        .unwrap_or_else(|| "main".to_string());
    let dir = cache_root().join(slug(&repo)).join(rev.replace('/', "_"));
    if dir.join("config.json").exists() && has_safetensors(&dir) {
        return Ok(dir);
    }
    if offline {
        bail!(
            "offline and repo {repo}@{rev} is not cached at {}",
            dir.display()
        );
    }
    fs::create_dir_all(&dir).with_context(|| format!("create cache dir {}", dir.display()))?;
    download_repo(&repo, &rev, &dir)?;
    if !has_safetensors(&dir) {
        bail!(
            "repo {repo}@{rev} has no *.safetensors files (downloaded to {})",
            dir.display()
        );
    }
    Ok(dir)
}

fn has_safetensors(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn auth_header() -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

fn get_bytes(url: &str) -> Result<Vec<u8>> {
    let mut req = ureq::get(url).timeout(std::time::Duration::from_secs(60));
    if let Some(token) = auth_header() {
        req = req.set("Authorization", &format!("Bearer {token}"));
    }
    let resp = req.call().map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .with_context(|| format!("read body of {url}"))?;
    Ok(buf)
}

fn download_repo(repo: &str, rev: &str, dir: &Path) -> Result<()> {
    eprintln!("lobotomy: resolving {repo}@{rev} …");
    let api_url = format!("{HF_API}/{repo}/revision/{rev}");
    let body = get_bytes(&api_url).with_context(|| format!("query HF API for {repo}"))?;
    let info: Value = serde_json::from_slice(&body).context("parse HF model info")?;
    let siblings = info["siblings"]
        .as_array()
        .context("HF model info has no siblings list")?;
    let mut wanted: Vec<String> = Vec::new();
    for s in siblings {
        let Some(name) = s["rfilename"].as_str() else {
            continue;
        };
        let keep = name == "config.json"
            || name == "model.safetensors.index.json"
            || name.ends_with(".safetensors")
            || name.starts_with("tokenizer")
            || name == "special_tokens_map.json"
            || name == "generation_config.json";
        if keep && !name.contains("..") {
            wanted.push(name.to_string());
        }
    }
    if wanted.is_empty() {
        bail!("repo {repo}@{rev} exposes no safetensors/config files");
    }
    for name in wanted {
        let dest = dir.join(&name);
        if dest.exists() {
            continue;
        }
        eprintln!("  fetching {name}");
        let url = format!("{HF_RESOLVE}/{repo}/resolve/{rev}/{name}");
        let bytes = get_bytes(&url)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, bytes).with_context(|| format!("write {}", dest.display()))?;
    }
    Ok(())
}
