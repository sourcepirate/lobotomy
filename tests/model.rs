use std::path::{Path, PathBuf};
use std::sync::Arc;

use lobotomy::edit::{EditOp, History};
use lobotomy::model::safetensors::{read_shard, write_file, DType, TensorSource};
use lobotomy::model::stats::{compute_stats, flatten2d, heatmap};
use lobotomy::model::{
    build_index, group_by_layer, relative_label, ModelIndex, TensorBuf, TensorStore,
};
use lobotomy::save::save_copy;
use std::collections::HashMap;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a small Mamba/Llama-flavoured fixture on disk.
fn fixture(dir: &Path) -> PathBuf {
    let root = dir.join("model");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), r#"{"model_type": "mamba"}"#).unwrap();
    let tensors: Vec<(String, String, Vec<usize>, TensorSource)> = vec![
        (
            "model.layers.3.mixer.in_proj.weight".into(),
            "F32".into(),
            vec![4, 2],
            TensorSource::Bytes(f32_bytes(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])),
        ),
        (
            "model.layers.3.mixer.A_log".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
        ),
        (
            "model.layers.3.mixer.D".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[1.0; 4])),
        ),
        (
            "model.layers.4.norm.weight".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[0.5; 4])),
        ),
        (
            "model.layers.23.mixer.D".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[2.0; 4])),
        ),
    ];
    write_file(&root.join("model.safetensors"), &tensors).unwrap();
    root
}

fn index(dir: &Path) -> ModelIndex {
    build_index(dir.to_str().unwrap(), None, false).unwrap()
}

#[test]
fn index_has_natural_order_and_relative_labels() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let idx = index(&root);
    assert_eq!(idx.arch, "mamba");
    assert_eq!(idx.tensors.len(), 5);

    let groups = group_by_layer(&idx.names());
    let names: Vec<&str> = groups.iter().map(|(g, _)| g.as_str()).collect();
    assert_eq!(
        names,
        ["model.layers.3", "model.layers.4", "model.layers.23"]
    );

    for (group, members) in &groups {
        let labels: Vec<String> = members.iter().map(|m| relative_label(group, m)).collect();
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            labels.len(),
            "duplicate labels in {group}: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l == "weight"),
            "bare `weight` label survived"
        );
    }
    let (_, members) = &groups[0];
    assert!(members
        .iter()
        .any(|m| relative_label("model.layers.3", m) == "mixer.in_proj.weight"));
}

#[test]
fn header_roundtrip_preserves_offsets_and_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let metas = read_shard(&root.join("model.safetensors")).unwrap();
    assert_eq!(metas.len(), 5);
    let q = metas
        .iter()
        .find(|m| m.name.ends_with("in_proj.weight"))
        .unwrap();
    assert_eq!(q.shape, vec![4, 2]);
    assert_eq!(q.shape_str(), "4×2");
    assert_eq!(q.numel(), 8);
    assert_eq!(q.data_len, 8 * 4);
}

#[test]
fn dtype_roundtrip_f16_bf16_f64() {
    let values = [0.125f32, -0.5, 3.25, 1024.0];
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
        let encoded = dtype.encode_f32(&values);
        assert_eq!(encoded.len(), values.len() * dtype.size());
        let decoded = dtype.decode_f32(&encoded);
        for (a, b) in values.iter().zip(decoded.iter()) {
            let tol = if dtype == DType::F16 || dtype == DType::BF16 {
                0.02 * a.abs().max(1.0)
            } else {
                1e-6
            };
            assert!((a - b).abs() <= tol, "{dtype:?}: {a} vs {b}");
        }
    }
}

#[test]
fn store_loads_tensors_and_reports_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let idx = index(&root);
    let mut store = TensorStore::new(1 << 20);
    let meta = idx.meta("model.layers.3.mixer.in_proj.weight").unwrap();
    let buf = store.get(meta).unwrap();
    assert_eq!(buf.shape, vec![4, 2]);
    assert_eq!(buf.data.len(), 8);
    assert!(store
        .cached("model.layers.3.mixer.in_proj.weight")
        .is_some());

    let stats = compute_stats(&buf);
    assert_eq!(stats.numel, 8);
    assert!(stats.min < -0.7 && stats.max > 0.6);
    assert!(stats.sparsity.abs() < 1e-6);
    assert_eq!(stats.hist.len(), 20);
}

#[test]
fn heatmap_is_one_row_for_vectors() {
    let buf = TensorBuf {
        name: "v".into(),
        shape: vec![32],
        dtype: DType::F32,
        data: (0..32).map(|i| i as f32).collect(),
    };
    let hm = heatmap(&buf, 16, 8, 0);
    assert_eq!(hm.grid.len(), 1, "1-D tensor must not be tiled to 16 rows");
    assert_eq!(hm.grid[0].len(), 8);
    assert_eq!(flatten2d(&buf), (1, 32));
}

#[test]
fn heatmap_pages_multidimensional_slices() {
    let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    let buf = TensorBuf {
        name: "m".into(),
        shape: vec![2, 3, 4],
        dtype: DType::F32,
        data,
    };
    let first = heatmap(&buf, 3, 4, 0);
    let second = heatmap(&buf, 3, 4, 1);
    assert_eq!(first.slice, Some(vec![0]));
    assert_eq!(second.slice, Some(vec![1]));
    assert_eq!(first.total_slices, 2);
    assert_ne!(first.grid, second.grid);
}

#[test]
fn edits_undo_and_redo() {
    let mut buf = TensorBuf {
        name: "t".into(),
        shape: vec![4],
        dtype: DType::F32,
        data: vec![1.0, -2.0, 0.5, 0.0],
    };
    let mut history = History::new();

    history.apply(&mut buf, EditOp::Scale { factor: 2.0 });
    assert_eq!(buf.data, vec![2.0, -4.0, 1.0, 0.0]);
    history.undo(&mut buf);
    assert_eq!(buf.data, vec![1.0, -2.0, 0.5, 0.0]);
    history.redo(&mut buf);
    assert_eq!(buf.data, vec![2.0, -4.0, 1.0, 0.0]);

    // destructive op needs a snapshot; undo must restore exactly
    history.apply(&mut buf, EditOp::Zero);
    assert_eq!(buf.data.iter().filter(|v| **v == 0.0).count(), 4);
    history.undo(&mut buf);
    assert_eq!(buf.data, vec![2.0, -4.0, 1.0, 0.0]);

    history.apply(&mut buf, EditOp::Prune { threshold: 1.5 });
    assert_eq!(buf.data, vec![2.0, -4.0, 0.0, 0.0]);
    history.undo(&mut buf);
    assert_eq!(buf.data, vec![2.0, -4.0, 1.0, 0.0]);

    history.apply(
        &mut buf,
        EditOp::SetCell {
            index: 2,
            value: 9.0,
        },
    );
    assert_eq!(buf.data[2], 9.0);
    history.undo(&mut buf);
    assert_eq!(buf.data[2], 1.0);

    assert!(history.is_dirty("t"));
    assert_eq!(history.log().len(), 4, "undo does not erase the diff log");
}

#[test]
fn clamp_and_noise_are_undoable() {
    let mut buf = TensorBuf {
        name: "t".into(),
        shape: vec![3],
        dtype: DType::F32,
        data: vec![-5.0, 0.0, 5.0],
    };
    let mut history = History::new();
    history.apply(&mut buf, EditOp::Clamp { lo: -1.0, hi: 1.0 });
    assert_eq!(buf.data, vec![-1.0, 0.0, 1.0]);
    history.undo(&mut buf);
    assert_eq!(buf.data, vec![-5.0, 0.0, 5.0]);

    let before = buf.data.clone();
    let seed = history.next_seed();
    history.apply(&mut buf, EditOp::Noise { sigma: 0.01, seed });
    assert_ne!(buf.data, before);
    history.undo(&mut buf);
    assert_eq!(buf.data, before);
}

#[test]
fn save_copy_streams_untouched_and_reencodes_edits() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let idx = index(&root);
    let mut store = TensorStore::new(1 << 20);
    let mut history = History::new();

    let meta = idx
        .meta("model.layers.3.mixer.in_proj.weight")
        .unwrap()
        .clone();
    let mut buf = (*store.get(&meta).unwrap()).clone();
    history.apply(&mut buf, EditOp::Scale { factor: 0.5 });
    store.insert(buf.clone());
    let mut edited = HashMap::new();
    edited.insert(meta.name.clone(), Arc::new(buf));

    let out = tmp.path().join("out");
    let report = save_copy(&idx, &edited, &history, &out).unwrap();
    assert_eq!(report.tensors, 5);
    assert_eq!(report.edits, 1);
    assert!(out.join("config.json").exists());
    assert_eq!(
        std::fs::read_to_string(out.join("lobotomy-diff.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );

    // edited tensor has halved values, untouched ones are byte-identical
    let saved = read_shard(&out.join("model.safetensors")).unwrap();
    assert_eq!(saved.len(), 5);
    let mut store2 = TensorStore::new(1 << 20);
    let reloaded = store2
        .get(
            saved
                .iter()
                .find(|m| m.name.ends_with("in_proj.weight"))
                .unwrap(),
        )
        .unwrap();
    assert!((reloaded.data[0] - 0.05).abs() < 1e-6);
    let a_log = store2
        .get(saved.iter().find(|m| m.name.ends_with("A_log")).unwrap())
        .unwrap();
    assert_eq!(a_log.data, vec![1.0, 2.0, 3.0, 4.0]);

    // never writes into the source
    let err = save_copy(&idx, &edited, &history, &root).unwrap_err();
    assert!(err.to_string().contains("refusing"));
}

#[test]
fn save_requires_edits() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let idx = index(&root);
    let edited = HashMap::new();
    let history = History::new();
    let err = save_copy(&idx, &edited, &history, &tmp.path().join("out")).unwrap_err();
    assert!(err.to_string().contains("no edits"));
}

#[test]
fn save_refuses_to_overwrite_an_existing_model_file() {
    let tmp = tempfile::tempdir().unwrap();
    let root = fixture(tmp.path());
    let idx = index(&root);
    let mut store = TensorStore::new(1 << 20);
    let meta = idx.tensors[0].clone();
    let mut buf = (*store.get(&meta).unwrap()).clone();
    let mut history = History::new();
    history.apply(&mut buf, EditOp::Scale { factor: 0.5 });
    store.insert(buf.clone());
    let mut edited = HashMap::new();
    edited.insert(meta.name.clone(), Arc::new(buf));

    let out = tmp.path().join("existing-output");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("model.safetensors"), b"keep me").unwrap();
    let err = save_copy(&idx, &edited, &history, &out).unwrap_err();
    assert!(err.to_string().contains("already contains"));
    assert_eq!(
        std::fs::read(out.join("model.safetensors")).unwrap(),
        b"keep me"
    );
}

#[test]
fn store_decodes_bf16_from_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("bf16");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    let values = [0.5f32, -1.25, 2.0, 0.0];
    let bytes = DType::BF16.encode_f32(&values);
    write_file(
        &root.join("model.safetensors"),
        &[(
            "w".to_string(),
            "BF16".to_string(),
            vec![4],
            TensorSource::Bytes(bytes.clone()),
        )],
    )
    .unwrap();
    let idx = build_index(root.to_str().unwrap(), None, false).unwrap();
    assert_eq!(idx.tensors[0].dtype, DType::BF16);
    let mut store = TensorStore::new(1 << 20);
    let buf = store.get(&idx.tensors[0]).unwrap();
    for (a, b) in values.iter().zip(buf.data.iter()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
    assert!(buf.is_editable());
}
