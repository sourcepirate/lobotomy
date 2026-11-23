use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lobotomy::edit::History;
use lobotomy::model::safetensors::{write_file, TensorSource};
use lobotomy::model::{build_index, ModelIndex};
use lobotomy::tui::{App, Mode, Tab};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn fixture(dir: &Path) -> ModelIndex {
    let root = dir.join("model");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
    let tensors: Vec<(String, String, Vec<usize>, TensorSource)> = vec![
        (
            "model.layers.3.mixer.in_proj.weight".into(),
            "F32".into(),
            vec![8, 4],
            TensorSource::Bytes(f32_bytes(
                &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )),
        ),
        (
            "model.layers.3.mixer.A_log".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
        ),
        (
            "model.layers.23.mixer.D".into(),
            "F32".into(),
            vec![4],
            TensorSource::Bytes(f32_bytes(&[2.0; 4])),
        ),
    ];
    write_file(&root.join("model.safetensors"), &tensors).unwrap();
    build_index(root.to_str().unwrap(), None, false).unwrap()
}

fn draw(app: &mut App) -> String {
    draw_at(app, 120, 32)
}

fn draw_at(app: &mut App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            out.push_str(buffer[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn confirm_zero(app: &mut App) {
    app.handle_key(key(KeyCode::Char('y')));
}

fn wait_for_save(app: &mut App) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while app.save_in_progress() && std::time::Instant::now() < deadline {
        app.poll_save();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    app.poll_save();
    assert!(!app.save_in_progress(), "save worker did not finish");
}

#[test]
fn renders_tree_labels_tabs_and_status() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    let text = draw(&mut app);

    assert!(text.contains("lobotomy"), "header missing");
    assert!(text.contains("model.layers.3"), "group missing");
    assert!(
        text.contains("mixer.in_proj.weight"),
        "relative label missing"
    );
    assert!(text.contains("[8×4]"), "shape annotation missing");
    assert!(text.contains("overview"), "tab bar missing");
    assert!(text.contains("NORMAL"), "mode badge missing");
    assert!(
        !text.contains("\u{FFFD}"),
        "rendering produced replacement chars"
    );
}

#[test]
fn navigation_and_tabs() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));

    let start = app.selected.clone();
    app.handle_key(key(KeyCode::Char('j')));
    app.handle_key(key(KeyCode::Char('k')));
    assert_eq!(app.selected, start);

    app.handle_key(key(KeyCode::Char('2')));
    assert_eq!(app.tab, Tab::Heatmap);
    let text = draw(&mut app);
    assert!(text.contains("heatmap"));

    app.handle_key(key(KeyCode::Char('3')));
    assert_eq!(app.tab, Tab::Cells);
    let text = draw(&mut app);
    assert!(text.contains("rows 0–"), "cells pager missing: {text}");

    app.handle_key(key(KeyCode::Char('?')));
    assert_eq!(app.tab, Tab::Help);
    let text = draw(&mut app);
    assert!(text.contains("navigate"));
    app.handle_key(key(KeyCode::Esc));
    assert_eq!(app.tab, Tab::Overview);
}

#[test]
fn collapse_and_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.tree.cursor = 0;
    app.handle_key(key(KeyCode::Char(' ')));
    assert!(app.tree.collapsed.contains("model.layers.3"));
    // the group is collapsed, so its tensor rows are gone
    assert!(!app.tree.rows.iter().any(|r| r.name.ends_with("A_log")));
    app.handle_key(key(KeyCode::Char(' ')));
    assert!(!app.tree.collapsed.contains("model.layers.3"));

    app.run_line("filter A_log");
    assert_eq!(app.tree.filter, "A_log");
    assert!(app
        .tree
        .rows
        .iter()
        .all(|r| r.name.contains("A_log") || r.count > 0));
    app.run_line("filter ");
}

#[test]
fn zero_edit_undo_redo_through_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    // select a tensor row
    app.tree.cursor = 1;
    app.handle_key(key(KeyCode::Enter));
    let name = app.selected.clone().unwrap();
    assert!(app.current_buf().is_some());

    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    let zeros = app
        .current_buf()
        .unwrap()
        .data
        .iter()
        .filter(|v| **v == 0.0)
        .count();
    assert_eq!(zeros, app.current_buf().unwrap().data.len());
    assert!(app.history.is_dirty(&name));

    app.handle_key(key(KeyCode::Char('u')));
    let restored = app
        .current_buf()
        .unwrap()
        .data
        .iter()
        .filter(|v| **v == 0.0)
        .count();
    assert!(restored < app.current_buf().unwrap().data.len());

    app.handle_key(key(KeyCode::Char('r')));
    let zeros = app
        .current_buf()
        .unwrap()
        .data
        .iter()
        .filter(|v| **v == 0.0)
        .count();
    assert_eq!(zeros, app.current_buf().unwrap().data.len());
}

#[test]
fn prompt_scale_applies_and_esc_cancels() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.tree.cursor = 1;
    app.handle_key(key(KeyCode::Enter));
    let before: Vec<f32> = app.current_buf().unwrap().data.clone();

    app.handle_key(key(KeyCode::Char('s')));
    assert_eq!(app.mode, Mode::Prompt);
    assert_eq!(app.prompt, "scale ");
    for c in "2.0".chars() {
        app.handle_key(key(KeyCode::Char(c)));
    }
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(app.mode, Mode::Normal);
    let after = app.current_buf().unwrap().data.clone();
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((b - a * 2.0).abs() < 1e-5);
    }

    // Esc cancels without applying
    app.handle_key(key(KeyCode::Char('s')));
    app.handle_key(key(KeyCode::Char('9')));
    app.handle_key(key(KeyCode::Esc));
    assert_eq!(app.mode, Mode::Normal);
    assert!(app.prompt.is_empty());
    let unchanged = app.current_buf().unwrap().data.clone();
    assert_eq!(unchanged, after);
}

#[test]
fn prompt_tab_completion_and_backspace() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.handle_key(key(KeyCode::Char('/')));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Backspace));
    assert!(app.prompt.is_empty());
    app.handle_key(key(KeyCode::Char('s')));
    app.handle_key(key(KeyCode::Tab));
    assert!(
        app.prompt.starts_with("scale")
            || app.prompt.starts_with("save")
            || app.prompt.starts_with("select")
            || app.prompt.starts_with("slice")
    );
}

#[test]
fn save_via_prompt_writes_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.tree.cursor = 1;
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    assert!(app.history.has_edits());

    let out = tmp.path().join("saved model");
    app.run_line(&format!("save \"{}\"", out.display()));
    wait_for_save(&mut app);
    assert!(out.join("model.safetensors").exists(), "{}", app.message);
    assert!(out.join("lobotomy-diff.jsonl").exists());
    assert!(!app.message_error, "{}", app.message);
}

#[test]
fn editing_refuses_non_float_dtype() {
    // create an integer tensor fixture; stats render but edits are rejected
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("m");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    let tensors: Vec<(String, String, Vec<usize>, TensorSource)> = vec![(
        "model.layers.0.ids".into(),
        "I64".into(),
        vec![2],
        TensorSource::Bytes(vec![0u8; 16]),
    )];
    write_file(&root.join("model.safetensors"), &tensors).unwrap();
    let idx = build_index(root.to_str().unwrap(), None, false).unwrap();
    let mut app = App::new(idx);
    assert!(app.message.contains("metadata-only"));
    assert!(draw(&mut app).contains("Value statistics"));
    app.tree.cursor = app
        .tree
        .rows
        .iter()
        .position(|r| r.kind == lobotomy::tui::tree::RowKind::Tensor)
        .unwrap();
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Char('z')));
    assert_eq!(app.mode, Mode::Normal);
    assert!(app.message_error, "integer tensor edit should be refused");
    assert!(!History::new().has_edits());
}

#[test]
fn undo_targets_the_edited_tensor_not_the_selected_one() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    // cursor starts on the selected tensor row
    assert!(app.tree.current().is_some());
    let edited = app.selected.clone().unwrap();

    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    assert!(app.current_buf().unwrap().data.iter().all(|v| *v == 0.0));

    // move to a different tensor
    app.tree.step_tensor(1);
    app.handle_key(key(KeyCode::Enter));
    let other = app.selected.clone().unwrap();
    assert_ne!(other, edited);
    let other_before = app.current_buf().unwrap().data.clone();

    // undo must revert the edited tensor, not touch the selected one
    app.handle_key(key(KeyCode::Char('u')));
    assert_eq!(app.selected.as_deref(), Some(edited.as_str()));
    assert!(app.current_buf().unwrap().data.iter().any(|v| *v != 0.0));
    let other_after = app
        .edited
        .get(&other)
        .map(|b| b.data.clone())
        .unwrap_or_else(|| app.store.cached(&other).unwrap().data.clone());
    assert_eq!(other_after, other_before);
}

#[test]
fn save_includes_edits_after_switching_tensors() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    let edited = app.selected.clone().unwrap();
    app.tree.step_tensor(1);
    app.handle_key(key(KeyCode::Enter));

    let out = tmp.path().join("out2");
    app.run_line(&format!("save {}", out.display()));
    wait_for_save(&mut app);
    assert!(!app.message_error, "{}", app.message);
    let metas = lobotomy::model::safetensors::read_shard(&out.join("model.safetensors")).unwrap();
    let mut store = lobotomy::model::TensorStore::new(1 << 20);
    let meta = metas.iter().find(|m| m.name == edited).unwrap();
    let buf = store.get(meta).unwrap();
    assert!(buf.data.iter().all(|v| *v == 0.0), "edit was lost on save");
}

#[test]
fn cell_grid_selects_and_edits_the_focused_cell() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.handle_key(key(KeyCode::Char('3')));
    app.handle_key(key(KeyCode::Right));
    assert_eq!(app.cell_col, 1);
    app.handle_key(key(KeyCode::Char('e')));
    assert!(app.prompt.starts_with("set 0,1 "));
    for c in "42".chars() {
        app.handle_key(key(KeyCode::Char(c)));
    }
    app.handle_key(key(KeyCode::Enter));
    let buf = app.current_buf().unwrap();
    assert_eq!(buf.data[1], 42.0);
    assert_eq!((app.cell_row, app.cell_col), (0, 1));
}

#[test]
fn filter_reports_no_matches_and_clears_hidden_selection() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.run_line("filter definitely-not-a-tensor");
    assert!(app.tree.rows.is_empty());
    assert!(app.selected.is_none());
    assert!(app.message.contains("no tensors match"));
    app.run_line("filter");
    assert!(!app.tree.rows.is_empty());
    assert!(app.selected.is_some());
}

#[test]
fn prompt_history_restores_commands_and_draft() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.open_prompt("filter ");
    for c in "A_log".chars() {
        app.handle_key(key(KeyCode::Char(c)));
    }
    app.handle_key(key(KeyCode::Enter));
    app.open_prompt("scale ");
    app.handle_key(key(KeyCode::Up));
    assert_eq!(app.prompt, "filter A_log");
    app.handle_key(key(KeyCode::Down));
    assert_eq!(app.prompt, "scale ");
}

#[test]
fn general_prompt_can_select_by_one_based_tensor_number() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    let expected = app.index.tensors[2].name.clone();
    app.handle_key(key(KeyCode::Char(':')));
    for c in "select 3".chars() {
        app.handle_key(key(KeyCode::Char(c)));
    }
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(app.selected.as_deref(), Some(expected.as_str()));
}

#[test]
fn quit_with_edits_requires_a_choice() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    app.handle_key(key(KeyCode::Char('q')));
    assert_eq!(app.mode, Mode::ConfirmQuit);
    assert!(!app.should_quit);
    app.handle_key(key(KeyCode::Esc));
    assert!(!app.should_quit);
    app.handle_key(key(KeyCode::Char('q')));
    app.handle_key(key(KeyCode::Char('d')));
    assert!(app.should_quit);
}

#[test]
fn invalid_cell_coordinates_are_rejected_not_wrapped() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.set_cell(99, 0, 1.0);
    assert!(app.message_error);
    assert!(!app.history.has_edits());
}

#[test]
fn narrow_terminal_explains_how_to_recover() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    let text = draw_at(&mut app, 60, 20);
    assert!(text.contains("too narrow"));
    assert!(text.contains("Resize"));
}

#[test]
fn save_runs_in_background_and_displays_completion() {
    let tmp = tempfile::tempdir().unwrap();
    let mut app = App::new(fixture(tmp.path()));
    app.handle_key(key(KeyCode::Char('z')));
    confirm_zero(&mut app);
    let out = tmp.path().join("background-save");
    app.run_line(&format!("save {}", out.display()));
    assert!(app.save_in_progress());
    app.apply_op(lobotomy::EditOp::Scale { factor: 2.0 });
    assert!(app.message.contains("wait for the current save"));
    app.handle_key(key(KeyCode::Char('q')));
    assert!(!app.should_quit);
    wait_for_save(&mut app);
    assert!(app.message.starts_with("saved "));
    assert!(out.join("model.safetensors").exists());
}
