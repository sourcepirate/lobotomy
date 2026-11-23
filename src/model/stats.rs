//! Statistics and downsampled views. All work is sampled so a 4M-element
//! tensor never freezes the UI.

use crate::model::TensorBuf;

pub const MAX_SAMPLE: usize = 200_000;
pub const HIST_BINS: usize = 20;

#[derive(Debug, Clone)]
pub struct Stats {
    pub numel: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub sparsity: f32,
    pub hist: Vec<u64>,
    pub edges: Vec<f32>,
}

pub fn compute_stats(buf: &TensorBuf) -> Stats {
    let data = &buf.data;
    let n = data.len();
    if n == 0 {
        return Stats {
            numel: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            sparsity: 0.0,
            hist: vec![0; HIST_BINS],
            edges: vec![0.0; HIST_BINS + 1],
        };
    }
    let stride = (n / MAX_SAMPLE).max(1);
    let sample: Vec<f32> = data.iter().step_by(stride).copied().collect();
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut zeros = 0usize;
    for &v in &sample {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
        sum += v as f64;
        sumsq += (v as f64) * (v as f64);
        if v == 0.0 {
            zeros += 1;
        }
    }
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 0.0;
    }
    let count = sample.len() as f64;
    let mean = (sum / count) as f32;
    let var = (sumsq / count) - (sum / count).powi(2);
    let std = var.max(0.0).sqrt() as f32;

    let mut hist = vec![0u64; HIST_BINS];
    let span = (max - min).max(1e-12);
    let edges: Vec<f32> = (0..=HIST_BINS)
        .map(|i| min + span * (i as f32 / HIST_BINS as f32))
        .collect();
    for &v in &sample {
        let mut idx = (((v - min) / span) * HIST_BINS as f32) as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= HIST_BINS {
            idx = HIST_BINS as isize - 1;
        }
        hist[idx as usize] += 1;
    }

    Stats {
        numel: n,
        min,
        max,
        mean,
        std,
        sparsity: zeros as f32 / count as f32,
        hist,
        edges,
    }
}

pub fn sparkline(values: &[u64], width: usize) -> String {
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || width == 0 {
        return String::new();
    }
    let bucket = values.len().div_ceil(width).max(1);
    let mut buckets: Vec<u64> = Vec::new();
    for chunk in values.chunks(bucket) {
        buckets.push(chunk.iter().sum());
    }
    let peak = buckets.iter().copied().max().unwrap_or(0).max(1);
    buckets
        .iter()
        .map(|&v| {
            let idx = ((v as f64 / peak as f64) * 8.0).round() as usize;
            BLOCKS[idx.min(8)]
        })
        .collect()
}

/// A pooled 2-D view of a tensor for rendering.
pub struct Heatmap {
    pub grid: Vec<Vec<f32>>,
    pub lo: f32,
    pub hi: f32,
    pub slice: Option<Vec<usize>>,
    pub total_slices: usize,
}

/// Pool a 2-D slice of `buf` into `rows × cols` cells.
///
/// For >2-D tensors, `head` selects the flattened leading-dim index, so the
/// user can page through attention heads / channels.
pub fn heatmap(buf: &TensorBuf, rows: usize, cols: usize, head: usize) -> Heatmap {
    let shape = &buf.shape;
    if buf.data.is_empty() || rows == 0 || cols == 0 {
        return Heatmap {
            grid: vec![vec![0.0; cols]; rows],
            lo: 0.0,
            hi: 0.0,
            slice: None,
            total_slices: 1,
        };
    }
    let mut total_slices = 1usize;
    let mut slice = None;
    let mut base = 0usize;
    if shape.len() > 2 {
        let lead = &shape[..shape.len() - 2];
        for d in lead {
            total_slices *= d;
        }
        let h = head % total_slices.max(1);
        let mut rem = h;
        let mut coords = vec![0usize; lead.len()];
        for i in (0..lead.len()).rev() {
            coords[i] = rem % lead[i];
            rem /= lead[i];
        }
        let strides = buf.strides();
        for (i, c) in coords.iter().enumerate() {
            base += c * strides[i];
        }
        slice = Some(coords);
    } else if shape.len() == 1 {
        // A vector is one row, not a tiled block.
        // rows is forced to 1 below.
    }
    let (r, c) = if shape.len() >= 2 {
        (shape[shape.len() - 2], shape[shape.len() - 1])
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (1, 1)
    };
    let rows = if shape.len() <= 1 { 1 } else { rows };
    let row_stride = if shape.len() >= 2 {
        buf.strides()[shape.len() - 2]
    } else {
        0
    };

    let mut grid = vec![vec![0.0f32; cols]; rows];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    // Sample at most 6×6 points per output cell to bound cost.
    let rs = (r as f64 / rows as f64).max(1.0);
    let cs = (c as f64 / cols as f64).max(1.0);
    let samples_r = rs.ceil().min(6.0) as usize;
    let samples_c = cs.ceil().min(6.0) as usize;
    for (i, out_row) in grid.iter_mut().enumerate() {
        for (j, cell) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            let mut n = 0usize;
            for sr in 0..samples_r {
                let rr = ((i as f64 * rs) as usize + sr).min(r.saturating_sub(1));
                for sc in 0..samples_c {
                    let cc = ((j as f64 * cs) as usize + sc).min(c.saturating_sub(1));
                    let idx = base + rr * row_stride + cc;
                    if let Some(v) = buf.data.get(idx) {
                        acc += *v as f64;
                        n += 1;
                    }
                }
            }
            let v = if n > 0 { (acc / n as f64) as f32 } else { 0.0 };
            *cell = v;
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() {
        lo = 0.0;
    }
    if !hi.is_finite() {
        hi = 0.0;
    }
    Heatmap {
        grid,
        lo,
        hi,
        slice,
        total_slices: total_slices.max(1),
    }
}

/// Flatten leading dims for the (row, col) view used by the cell grid.
pub fn flatten2d(buf: &TensorBuf) -> (usize, usize) {
    match buf.shape.len() {
        0 => (1, 1),
        1 => (1, buf.shape[0]),
        n => (buf.data.len() / buf.shape[n - 1].max(1), buf.shape[n - 1]),
    }
}
