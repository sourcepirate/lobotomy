//! safetensors reading and writing.
//!
//! Implemented directly rather than through a wrapper so we keep raw byte
//! offsets per tensor: saving can then stream untouched tensors straight
//! from the mmap instead of decoding and re-encoding gigabytes.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use serde_json::{json, Value};

/// Element type of a tensor as stored on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
    Other(String),
}

impl DType {
    pub fn parse(s: &str) -> Self {
        match s {
            "F64" => DType::F64,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "I16" => DType::I16,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "BOOL" => DType::Bool,
            other => DType::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::Bool => "BOOL",
            DType::Other(s) => s,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            _ => 1,
        }
    }

    /// Dtypes we can round-trip through an f32 working buffer.
    pub fn is_editable(&self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::F16 | DType::BF16)
    }

    /// Decode raw little-endian bytes into an f32 working buffer.
    pub fn decode_f32(&self, bytes: &[u8]) -> Vec<f32> {
        match self {
            DType::F64 => bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Encode an f32 working buffer back into this dtype.
    pub fn encode_f32(&self, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * self.size());
        match self {
            DType::F64 => {
                for v in values {
                    out.extend_from_slice(&(*v as f64).to_le_bytes());
                }
            }
            DType::F32 => {
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::F16 => {
                for v in values {
                    out.extend_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
                }
            }
            DType::BF16 => {
                for v in values {
                    out.extend_from_slice(&bf16::from_f32(*v).to_bits().to_le_bytes());
                }
            }
            _ => {}
        }
        out
    }
}

/// One tensor's location inside a shard: header metadata plus byte range.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub shard: PathBuf,
    pub data_start: u64,
    pub data_len: u64,
}

impl TensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// `[4096×4096]` style label.
    pub fn shape_str(&self) -> String {
        if self.shape.is_empty() {
            "scalar".to_string()
        } else {
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("×")
        }
    }
}

/// Shape::default-like helper.
fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Parse only the JSON header of a safetensors file.
pub fn read_header(path: &Path) -> Result<(Value, u64)> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(f);
    let mut len_buf = [0u8; 8];
    reader
        .read_exact(&mut len_buf)
        .with_context(|| format!("read header length of {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        bail!(
            "implausible safetensors header length {header_len} in {}",
            path.display()
        );
    }
    let mut header = vec![0u8; header_len as usize];
    reader.read_exact(&mut header)?;
    let value: Value = serde_json::from_slice(&header)
        .with_context(|| format!("parse safetensors header of {}", path.display()))?;
    Ok((value, header_len))
}

/// Read every tensor entry from one shard (no tensor data is loaded).
pub fn read_shard(path: &Path) -> Result<Vec<TensorMeta>> {
    let (header, header_len) = read_header(path)?;
    let obj = match header.as_object() {
        Some(o) => o,
        None => bail!("safetensors header is not an object: {}", path.display()),
    };
    let data_base = 8 + header_len;
    let mut out = Vec::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = DType::parse(info["dtype"].as_str().unwrap_or("UNKNOWN"));
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let offsets = info["data_offsets"].as_array();
        let (begin, end) = match offsets {
            Some(o) if o.len() == 2 => (o[0].as_u64().unwrap_or(0), o[1].as_u64().unwrap_or(0)),
            _ => bail!("tensor {name} has no data_offsets"),
        };
        let expected = product(&shape) * dtype.size();
        if (end - begin) as usize != expected {
            bail!(
                "tensor {name} in {} declares {} bytes but shape/dtype imply {}",
                path.display(),
                end - begin,
                expected
            );
        }
        out.push(TensorMeta {
            name: name.clone(),
            shape,
            dtype,
            shard: path.to_path_buf(),
            data_start: data_base + begin,
            data_len: end - begin,
        });
    }
    Ok(out)
}

/// Memory-mapped shard, opened on demand and cached by the caller.
pub struct Shard {
    mmap: Mmap,
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        Ok(Shard { mmap })
    }

    pub fn bytes(&self, meta: &TensorMeta) -> Result<&[u8]> {
        let start = meta.data_start as usize;
        let end = start + meta.data_len as usize;
        self.mmap.get(start..end).with_context(|| {
            format!(
                "range {}..{end} out of bounds in {}",
                start,
                meta.shard.display()
            )
        })
    }
}

/// Write a fresh safetensors file.
///
/// `tensors` is ordered; untouched tensors provide `Raw` byte ranges to copy
/// directly from their shard so we never decode them.
pub enum TensorSource {
    Raw {
        shard: PathBuf,
        start: u64,
        len: u64,
    },
    Bytes(Vec<u8>),
}

pub fn write_file(
    path: &Path,
    tensors: &[(String, String, Vec<usize>, TensorSource)],
) -> Result<()> {
    write_file_with_progress(path, tensors, |_, _, _| {})
}

pub fn write_file_with_progress(
    path: &Path,
    tensors: &[(String, String, Vec<usize>, TensorSource)],
    mut progress: impl FnMut(u64, u64, &str),
) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset: u64 = 0;
    for (name, dtype, shape, source) in tensors {
        let len = match source {
            TensorSource::Raw { len, .. } => *len,
            TensorSource::Bytes(b) => b.len() as u64,
        };
        header.insert(
            name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + len],
            }),
        );
        offset += len;
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header))?;
    let total_bytes = offset;
    let mut out = std::io::BufWriter::new(
        File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    out.write_all(&header_bytes)?;

    // group raw copies per shard so each file is opened once
    let mut open_shards: BTreeMap<PathBuf, File> = BTreeMap::new();
    let mut bytes_done = 0u64;
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for (name, _dtype, _shape, source) in tensors {
        match source {
            TensorSource::Bytes(b) => {
                for chunk in b.chunks(buffer.len()) {
                    out.write_all(chunk)?;
                    bytes_done += chunk.len() as u64;
                    progress(bytes_done, total_bytes, name);
                }
            }
            TensorSource::Raw { shard, start, len } => {
                let f = match open_shards.get_mut(shard) {
                    Some(f) => f,
                    None => {
                        let f = File::open(shard)
                            .with_context(|| format!("reopen {} for copy", shard.display()))?;
                        open_shards.insert(shard.clone(), f);
                        open_shards.get_mut(shard).unwrap()
                    }
                };
                f.seek(SeekFrom::Start(*start))?;
                let mut remaining = *len;
                while remaining > 0 {
                    let chunk_len = remaining.min(buffer.len() as u64) as usize;
                    let read = f.read(&mut buffer[..chunk_len])?;
                    if read == 0 {
                        bail!(
                            "short copy for tensor {name}: {} of {len} bytes",
                            len - remaining
                        );
                    }
                    out.write_all(&buffer[..read])?;
                    remaining -= read as u64;
                    bytes_done += read as u64;
                    progress(bytes_done, total_bytes, name);
                }
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Verify the file re-parses and exposes exactly `expected` names.
pub fn verify_file(path: &Path, expected: &[String]) -> Result<()> {
    let meta = read_shard(path)?;
    let got: Vec<String> = meta.into_iter().map(|m| m.name).collect();
    let mut want = expected.to_vec();
    want.sort();
    let mut got_sorted = got.clone();
    got_sorted.sort();
    if want != got_sorted {
        bail!(
            "round-trip key mismatch in {}: expected {} tensors, found {}",
            path.display(),
            want.len(),
            got_sorted.len()
        );
    }
    Ok(())
}
