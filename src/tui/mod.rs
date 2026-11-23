//! Application state, key handling and the event loop.

pub mod render;
pub mod tree;
pub mod widgets;

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;

use crate::edit::{EditOp, History};
use crate::model::stats::{compute_stats, flatten2d, Stats};
use crate::model::{Meta, ModelIndex, TensorBuf, TensorStore};
use crate::save::{default_save_dir, save_copy_with_progress, SaveReport};

use tree::{RowKind, TreeView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Heatmap,
    Cells,
    Help,
}

impl Tab {
    pub fn index(self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::Heatmap => 1,
            Tab::Cells => 2,
            Tab::Help => 3,
        }
    }
    pub fn names() -> [&'static str; 4] {
        ["1 overview", "2 heatmap", "3 cells", "? help"]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Prompt,
    ConfirmZero,
    ConfirmQuit,
}

pub const PAGE: usize = 16;
pub const CELL_COLS: usize = 8;
pub const HEAT_ROWS: usize = 16;
const CACHE_BUDGET: usize = 512 * 1024 * 1024;

enum SaveEvent {
    Progress {
        bytes_done: u64,
        total_bytes: u64,
        current: String,
    },
    Finished(Result<SaveReport, String>),
}

pub struct App {
    pub index: ModelIndex,
    pub store: TensorStore,
    pub history: History,
    pub tree: TreeView,
    pub tab: Tab,
    pub mode: Mode,
    pub selected: Option<String>,
    pub prompt: String,
    pub prompt_cursor: usize,
    pub prompt_history_cursor: Option<usize>,
    pub prompt_draft: String,
    pub message: String,
    pub message_error: bool,
    pub slice_head: usize,
    pub cell_page: usize,
    pub should_quit: bool,
    pub prompt_history: VecDeque<String>,
    pub cell_row: usize,
    pub cell_col: usize,
    pending_zero: Option<String>,
    save_receiver: Option<Receiver<SaveEvent>>,
    save_running: bool,
    quit_after_save: bool,
    /// Edited tensors kept independently of the LRU cache so saving never
    /// depends on what is currently resident.
    pub edited: HashMap<String, Arc<TensorBuf>>,
    /// Cached stats for the selected tensor, invalidated on edit.
    stats_cache: Option<(String, usize, Stats)>,
}

impl App {
    pub fn new(index: ModelIndex) -> Self {
        let mut app = Self {
            index,
            store: TensorStore::new(CACHE_BUDGET),
            history: History::new(),
            tree: TreeView::new(),
            tab: Tab::Overview,
            mode: Mode::Normal,
            selected: None,
            prompt: String::new(),
            prompt_cursor: 0,
            prompt_history_cursor: None,
            prompt_draft: String::new(),
            message: String::new(),
            message_error: false,
            slice_head: 0,
            cell_page: 0,
            should_quit: false,
            prompt_history: VecDeque::new(),
            cell_row: 0,
            cell_col: 0,
            pending_zero: None,
            save_receiver: None,
            save_running: false,
            quit_after_save: false,
            edited: HashMap::new(),
            stats_cache: None,
        };
        app.rebuild();
        let first = app
            .tree
            .rows
            .iter()
            .find(|r| {
                r.kind == RowKind::Tensor
                    && r.name
                        .split('.')
                        .collect::<Vec<_>>()
                        .windows(2)
                        .any(|pair| {
                            matches!(pair[0], "layers" | "layer" | "blocks" | "block" | "h")
                                && !pair[1].is_empty()
                                && pair[1].chars().all(|c| c.is_ascii_digit())
                        })
            })
            .or_else(|| app.tree.rows.iter().find(|r| r.kind == RowKind::Tensor))
            .map(|r| r.name.clone())
            .or_else(|| app.tree.rows.first().map(|r| r.name.clone()));
        if let Some(name) = first {
            app.select(&name);
            app.rebuild(); // move the cursor onto the selection
        }
        app
    }

    pub fn rebuild(&mut self) {
        let previous_selection = self.selected.clone();
        let history = &self.history;
        self.tree
            .rebuild(&self.index, &|name| history.is_dirty(name));
        // Keep the visible selection and the detail pane aligned after filtering.
        if let Some(sel) = &self.selected {
            if let Some(i) = self
                .tree
                .rows
                .iter()
                .position(|r| r.kind == RowKind::Tensor && &r.name == sel)
            {
                self.tree.cursor = i;
            } else if !self.tree.filter.is_empty() {
                self.selected = self
                    .tree
                    .rows
                    .iter()
                    .find(|r| r.kind == RowKind::Tensor)
                    .map(|r| r.name.clone());
            }
        } else if let Some(row) = self.tree.rows.iter().find(|r| r.kind == RowKind::Tensor) {
            self.selected = Some(row.name.clone());
        }
        if let Some(name) = &self.selected {
            if let Some(i) = self
                .tree
                .rows
                .iter()
                .position(|r| r.kind == RowKind::Tensor && &r.name == name)
            {
                self.tree.cursor = i;
            }
        }
        if self.selected != previous_selection {
            if let Some(name) = self.selected.clone() {
                self.select(&name);
            } else {
                self.stats_cache = None;
            }
        }
    }

    pub fn select(&mut self, name: &str) {
        self.selected = Some(name.to_string());
        self.slice_head = 0;
        self.cell_page = 0;
        self.cell_row = 0;
        self.cell_col = 0;
        self.stats_cache = None;
        self.load_selected();
    }

    fn load_selected(&mut self) {
        let Some(name) = self.selected.clone() else {
            return;
        };
        let Some(meta) = self.index.meta(&name).cloned() else {
            return;
        };
        if !meta.dtype.is_editable() {
            self.message = format!(
                "{} · {} · values are metadata-only (read-only)",
                short(&name),
                meta.dtype.as_str()
            );
            self.message_error = false;
            return;
        }
        match self.store.get(&meta) {
            Ok(_) => {
                self.message = format!("loaded {} [{}]", short(&name), meta.shape_str());
                self.message_error = false;
            }
            Err(e) => {
                self.message = format!("load failed: {e}");
                self.message_error = true;
            }
        }
    }

    pub fn current_meta(&self) -> Option<&Meta> {
        self.selected.as_deref().and_then(|n| self.index.meta(n))
    }

    pub fn current_buf(&self) -> Option<Arc<TensorBuf>> {
        self.selected.as_deref().and_then(|n| self.store.cached(n))
    }

    pub fn stats(&mut self) -> Option<Stats> {
        let name = self.selected.clone()?;
        let version = self.history.log().len();
        if let Some((cached_name, cached_version, stats)) = &self.stats_cache {
            if cached_name == &name && *cached_version == version {
                return Some(stats.clone());
            }
        }
        let buf = self.current_buf()?;
        let stats = compute_stats(&buf);
        self.stats_cache = Some((name, version, stats.clone()));
        Some(stats)
    }

    // ------------------------------------------------------------- editing

    fn edit_current(&mut self, op: EditOp) -> Option<String> {
        if self.save_running {
            self.set_error("wait for the current save to finish before editing".into());
            return None;
        }
        let name = self.selected.clone()?;
        let meta = self.index.meta(&name)?.clone();
        let arc = match self.store.get(&meta) {
            Ok(a) => a,
            Err(e) => {
                self.set_error(format!("load failed: {e}"));
                return None;
            }
        };
        if !arc.is_editable() {
            self.set_error(format!(
                "{} is {} (read-only)",
                short(&name),
                arc.dtype.as_str()
            ));
            return None;
        }
        let mut buf: TensorBuf = (*arc).clone();
        let msg = self.history.apply(&mut buf, op);
        self.edited.insert(name, Arc::new(buf.clone()));
        self.store.insert(buf);
        self.stats_cache = None;
        self.rebuild();
        self.message_error = false;
        self.message = msg.clone();
        Some(msg)
    }

    pub fn apply_op(&mut self, op: EditOp) {
        self.edit_current(op);
    }

    pub fn set_cell(&mut self, row: usize, col: usize, value: f32) {
        let Some(buf) = self.current_buf() else {
            self.set_error("nothing selected".into());
            return;
        };
        let (rows, cols) = flatten2d(&buf);
        if rows == 0 || cols == 0 {
            self.set_error("empty tensor".into());
            return;
        }
        if row >= rows || col >= cols {
            self.set_error(format!(
                "cell ({row}, {col}) is outside this tensor; valid rows are 0–{} and columns are 0–{}",
                rows - 1,
                cols - 1
            ));
            return;
        }
        if !value.is_finite() {
            self.set_error("cell value must be a finite number".into());
            return;
        }
        self.cell_row = row;
        self.cell_col = col;
        self.cell_page = row / PAGE;
        let index = row * cols + col;
        self.edit_current(EditOp::SetCell { index, value });
    }

    fn request_zero(&mut self) {
        if self.save_running {
            self.set_error("wait for the current save to finish before editing".into());
            return;
        }
        let Some(name) = self.selected.clone() else {
            self.set_error("select a tensor before zeroing it".into());
            return;
        };
        if let Some(meta) = self.index.meta(&name) {
            if !meta.dtype.is_editable() {
                self.set_error(format!(
                    "{} is {} (read-only)",
                    short(&name),
                    meta.dtype.as_str()
                ));
                return;
            }
        }
        self.pending_zero = Some(name.clone());
        self.mode = Mode::ConfirmZero;
        self.message = format!("Zero all values in {}? This can be undone.", short(&name));
        self.message_error = false;
    }

    fn confirm_zero(&mut self, yes: bool) {
        if yes {
            if let Some(name) = self.pending_zero.take() {
                if self.selected.as_deref() == Some(name.as_str()) {
                    self.apply_op(EditOp::Zero);
                }
            }
        } else {
            self.pending_zero = None;
            self.message = "zero cancelled".into();
            self.message_error = false;
        }
        self.mode = Mode::Normal;
    }

    pub fn undo(&mut self) {
        if self.save_running {
            self.set_error("wait for the current save to finish before undoing".into());
            return;
        }
        let Some(name) = self.history.undo_tensor().map(str::to_string) else {
            self.message = "nothing to undo".into();
            self.message_error = false;
            return;
        };
        let target = self.apply_history_op(&name, true);
        self.finish_history_op(&name, target);
    }

    pub fn redo(&mut self) {
        if self.save_running {
            self.set_error("wait for the current save to finish before redoing".into());
            return;
        }
        let Some(name) = self.history.redo_tensor().map(str::to_string) else {
            self.message = "nothing to redo".into();
            self.message_error = false;
            return;
        };
        let target = self.apply_history_op(&name, false);
        self.finish_history_op(&name, target);
    }

    /// Undo/redo act on the tensor recorded in the edit, not on whatever
    /// happens to be selected.
    fn apply_history_op(&mut self, name: &str, is_undo: bool) -> Option<String> {
        let meta = self.index.meta(name)?.clone();
        let arc = match self.store.get(&meta) {
            Ok(a) => a,
            Err(e) => {
                self.set_error(format!("load failed: {e}"));
                return None;
            }
        };
        let mut buf: TensorBuf = (*arc).clone();
        let msg = if is_undo {
            self.history.undo(&mut buf)
        } else {
            self.history.redo(&mut buf)
        };
        self.edited.insert(name.to_string(), Arc::new(buf.clone()));
        self.store.insert(buf);
        Some(msg)
    }

    fn finish_history_op(&mut self, name: &str, msg: Option<String>) {
        let Some(msg) = msg else { return };
        self.stats_cache = None;
        if self.selected.as_deref() != Some(name) {
            self.selected = Some(name.to_string());
        }
        self.rebuild();
        self.message_error = false;
        self.message = msg;
    }

    pub fn save(&mut self, out: Option<String>) {
        if self.save_running {
            self.set_error("a save is already in progress".into());
            return;
        }
        if !self.history.has_edits() {
            self.set_error("no edits yet — nothing to save".into());
            return;
        }
        let dest = out
            .filter(|s| !s.trim().is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| default_save_dir(&self.index));
        let (sender, receiver) = mpsc::sync_channel(1);
        let index = self.index.clone();
        let edited = self.edited.clone();
        let dirty = self.history.dirty_tensors();
        let log = self.history.log().to_vec();
        let worker_dest = dest.clone();
        std::thread::spawn(move || {
            let result = save_copy_with_progress(
                &index,
                &edited,
                &dirty,
                &log,
                &worker_dest,
                |bytes_done, total_bytes, current| {
                    let _ = sender.try_send(SaveEvent::Progress {
                        bytes_done,
                        total_bytes,
                        current: current.to_string(),
                    });
                },
            )
            .map_err(|error| error.to_string());
            // A final result must not be dropped if the last progress update
            // is still occupying the bounded channel.
            let _ = sender.send(SaveEvent::Finished(result));
        });
        self.save_receiver = Some(receiver);
        self.save_running = true;
        self.message = format!("starting save to {} …", dest.display());
        self.message_error = false;
    }

    pub fn poll_save(&mut self) {
        let Some(receiver) = &self.save_receiver else {
            return;
        };
        let mut finished = None;
        loop {
            match receiver.try_recv() {
                Ok(SaveEvent::Progress {
                    bytes_done,
                    total_bytes,
                    current,
                }) => {
                    let percent = bytes_done
                        .saturating_mul(100)
                        .checked_div(total_bytes)
                        .unwrap_or(100)
                        .min(100);
                    self.message = format!(
                        "saving {percent}% · {:.1}/{:.1} MiB · {current}",
                        bytes_done as f64 / (1024.0 * 1024.0),
                        total_bytes as f64 / (1024.0 * 1024.0)
                    );
                    self.message_error = false;
                }
                Ok(SaveEvent::Finished(result)) => {
                    finished = Some(result);
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = Some(Err("save worker stopped unexpectedly".into()));
                    break;
                }
            }
        }
        if let Some(result) = finished {
            self.save_receiver = None;
            self.save_running = false;
            match result {
                Ok(report) => {
                    self.message = format!(
                        "saved {} tensors · {} edit operations → {}",
                        report.tensors,
                        report.edits,
                        report.dest.display()
                    );
                    self.message_error = false;
                    if self.quit_after_save {
                        self.should_quit = true;
                    }
                }
                Err(error) => {
                    self.message = format!("save failed: {error}");
                    self.message_error = true;
                    self.quit_after_save = false;
                }
            }
        }
    }

    pub fn save_in_progress(&self) -> bool {
        self.save_running
    }

    fn set_error(&mut self, msg: String) {
        self.message = msg;
        self.message_error = true;
    }

    // ------------------------------------------------------------- prompt

    pub fn open_prompt(&mut self, prefix: &str) {
        self.mode = Mode::Prompt;
        self.prompt = prefix.to_string();
        self.prompt_cursor = self.prompt.chars().count();
        self.prompt_history_cursor = None;
        self.prompt_draft.clear();
    }

    fn close_prompt(&mut self) {
        self.mode = Mode::Normal;
        self.prompt.clear();
        self.prompt_cursor = 0;
    }

    fn prompt_push(&mut self, c: char) {
        let byte = char_to_byte(&self.prompt, self.prompt_cursor);
        self.prompt.insert(byte, c);
        self.prompt_cursor += 1;
    }

    fn prompt_backspace(&mut self) {
        if self.prompt_cursor == 0 {
            return;
        }
        let end = char_to_byte(&self.prompt, self.prompt_cursor);
        let start = char_to_byte(&self.prompt, self.prompt_cursor - 1);
        self.prompt.replace_range(start..end, "");
        self.prompt_cursor -= 1;
    }

    fn prompt_left(&mut self) {
        self.prompt_cursor = self.prompt_cursor.saturating_sub(1);
    }

    fn prompt_right(&mut self) {
        self.prompt_cursor = (self.prompt_cursor + 1).min(self.prompt.chars().count());
    }

    fn complete(&mut self) {
        const COMMANDS: [&str; 12] = [
            "scale", "noise", "prune", "clamp", "set", "save", "select", "filter", "slice", "page",
            "undo", "redo",
        ];
        let text = self.prompt.clone();
        let word_start = text.rfind(' ').map(|i| i + 1).unwrap_or(0);
        let head = &text[..word_start];
        let word = &text[word_start..];
        let candidates: Vec<String> = if head.trim().is_empty() {
            COMMANDS
                .iter()
                .filter(|c| c.starts_with(word))
                .map(|c| c.to_string())
                .collect()
        } else if matches!(head.trim(), "filter" | "select") {
            self.index
                .names()
                .into_iter()
                .filter(|n| n.to_lowercase().contains(&word.to_lowercase()))
                .collect()
        } else {
            Vec::new()
        };
        if candidates.is_empty() {
            return;
        }
        // Complete to the longest common prefix, or use the first full match.
        let completion = common_prefix(&candidates).unwrap_or_else(|| candidates[0].clone());
        let completion = if completion.len() > word.len() {
            completion
        } else {
            candidates[0].clone()
        };
        self.prompt = format!("{head}{completion} ");
        self.prompt_cursor = self.prompt.chars().count();
    }

    fn submit_prompt(&mut self) {
        let line = self.prompt.trim().to_string();
        self.close_prompt();
        if line.is_empty() {
            return;
        }
        if self.prompt_history.back() != Some(&line) {
            self.prompt_history.push_back(line.clone());
            if self.prompt_history.len() > 50 {
                self.prompt_history.pop_front();
            }
        }
        self.prompt_history_cursor = None;
        self.prompt_draft.clear();
        self.run_line(&line);
    }

    fn prompt_history_move(&mut self, delta: isize) {
        if self.prompt_history.is_empty() {
            return;
        }
        let len = self.prompt_history.len();
        let current = self.prompt_history_cursor.unwrap_or(len);
        let next = (current as isize + delta).clamp(0, len as isize) as usize;
        if self.prompt_history_cursor.is_none() && next < len {
            self.prompt_draft = self.prompt.clone();
        }
        if next == len {
            self.prompt_history_cursor = None;
            self.prompt = self.prompt_draft.clone();
        } else {
            self.prompt_history_cursor = Some(next);
            self.prompt = self.prompt_history[next].clone();
        }
        self.prompt_cursor = self.prompt.chars().count();
    }

    pub fn run_line(&mut self, line: &str) {
        let line = line.trim();
        let mut parts = line.split_whitespace();
        let Some(cmd) = parts.next() else { return };
        let args: Vec<&str> = parts.collect();
        let num = |i: usize| {
            args.get(i)
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|v| v.is_finite())
        };
        match cmd {
            "scale" => match num(0) {
                Some(factor) => self.apply_op(EditOp::Scale { factor }),
                None => self.set_error("usage: scale <factor>".into()),
            },
            "noise" => match num(0) {
                Some(sigma) if sigma >= 0.0 => {
                    let seed = self.history.next_seed();
                    self.apply_op(EditOp::Noise { sigma, seed });
                }
                Some(_) => self.set_error("noise sigma must be 0 or greater".into()),
                None => self.set_error("usage: noise <sigma>".into()),
            },
            "zero" => self.request_zero(),
            "prune" => match num(0) {
                Some(threshold) if threshold >= 0.0 => self.apply_op(EditOp::Prune { threshold }),
                Some(_) => self.set_error("prune threshold must be 0 or greater".into()),
                None => self.set_error("usage: prune <threshold>".into()),
            },
            "clamp" => match (num(0), num(1)) {
                (Some(lo), Some(hi)) if lo <= hi => self.apply_op(EditOp::Clamp { lo, hi }),
                (Some(_), Some(_)) => {
                    self.set_error("clamp lower bound must not exceed upper bound".into())
                }
                _ => self.set_error("usage: clamp <lo> <hi>".into()),
            },
            "set" => {
                let (Some(pos), Some(v)) = (args.first(), num(1)) else {
                    return self.set_error("usage: set <row>,<col> <value>".into());
                };
                let Some((r, c)) = pos.split_once(',') else {
                    return self.set_error("usage: set <row>,<col> <value>".into());
                };
                match (r.parse::<usize>(), c.parse::<usize>()) {
                    (Ok(r), Ok(c)) => self.set_cell(r, c, v),
                    _ => self.set_error("usage: set <row>,<col> <value>".into()),
                }
            }
            "save" => self.save(save_path_argument(line)),
            "select" => {
                let query = args.join(" ");
                let found = if let Ok(n) = query.parse::<usize>() {
                    let found = n
                        .checked_sub(1)
                        .and_then(|i| self.index.tensors.get(i))
                        .map(|m| m.name.clone());
                    if found.is_none() {
                        self.set_error(format!(
                            "tensor number must be between 1 and {}",
                            self.index.tensors.len()
                        ));
                    }
                    found
                } else if self.index.meta(&query).is_some() {
                    Some(query.clone())
                } else {
                    let matches: Vec<_> = self
                        .index
                        .tensors
                        .iter()
                        .filter(|m| m.name.contains(&query))
                        .map(|m| m.name.clone())
                        .collect();
                    if matches.len() == 1 {
                        matches.into_iter().next()
                    } else {
                        self.set_error(if matches.is_empty() {
                            format!("no tensor matches {query:?}")
                        } else {
                            format!("{} tensors match {query:?}; refine the name", matches.len())
                        });
                        None
                    }
                };
                if let Some(name) = found {
                    self.tree.filter.clear();
                    if let Some((group, _)) =
                        crate::model::group_by_layer(std::slice::from_ref(&name)).first()
                    {
                        self.tree.collapsed.remove(group);
                    }
                    self.select(&name);
                    self.rebuild();
                }
            }
            "filter" => {
                self.tree.filter = args.join(" ");
                self.tree.cursor = 0;
                self.rebuild();
                self.message = if self.tree.filter.is_empty() {
                    format!(
                        "showing all {} tensors",
                        self.tree
                            .rows
                            .iter()
                            .filter(|r| r.kind == RowKind::Tensor)
                            .count()
                    )
                } else if self.tree.rows.iter().any(|r| r.kind == RowKind::Tensor) {
                    format!(
                        "filter: {} · {} matching tensors",
                        self.tree.filter,
                        self.tree
                            .rows
                            .iter()
                            .filter(|r| r.kind == RowKind::Tensor)
                            .count()
                    )
                } else {
                    format!(
                        "no tensors match {:?} · clear the filter with / then Enter",
                        self.tree.filter
                    )
                };
                self.message_error = false;
            }
            "slice" => self.nudge_slice(args.first().copied()),
            "page" => self.nudge_page(args.first().copied()),
            "undo" => self.undo(),
            "redo" => self.redo(),
            other => self.set_error(format!("unknown command {other:?} — press ? for help")),
        }
    }

    fn nudge_slice(&mut self, arg: Option<&str>) {
        let total = self
            .current_buf()
            .map(|buf| {
                buf.shape
                    .iter()
                    .take(buf.shape.len().saturating_sub(2))
                    .product::<usize>()
                    .max(1)
            })
            .unwrap_or(1);
        if total == 1 && matches!(arg, Some("+") | Some("-") | None) {
            self.slice_head = 0;
            self.message = "this tensor has a single 2D view; there are no other slices".into();
            self.message_error = false;
            return;
        }
        match arg {
            Some("+") | None => self.slice_head = (self.slice_head + 1).min(total - 1),
            Some("-") => self.slice_head = self.slice_head.saturating_sub(1),
            Some(n) => match n.parse::<usize>() {
                Ok(v) if (1..=total).contains(&v) => self.slice_head = v - 1,
                _ => {
                    self.set_error(format!("slice must be between 1 and {total}"));
                    return;
                }
            },
        }
        self.message = format!("slice {}/{}", self.slice_head + 1, total);
        self.message_error = false;
    }

    fn nudge_page(&mut self, arg: Option<&str>) {
        let Some(buf) = self.current_buf() else {
            self.set_error("select a tensor before changing cell pages".into());
            return;
        };
        let (rows, _) = flatten2d(&buf);
        let pages = rows.div_ceil(PAGE).max(1);
        match arg {
            Some("+") | None => self.cell_page = (self.cell_page + 1).min(pages - 1),
            Some("-") => self.cell_page = self.cell_page.saturating_sub(1),
            Some(n) => match n.parse::<usize>() {
                Ok(v) if (1..=pages).contains(&v) => self.cell_page = v - 1,
                _ => {
                    self.set_error(format!("page must be between 1 and {pages}"));
                    return;
                }
            },
        }
        if rows > 0 {
            self.cell_row = (self.cell_page * PAGE).min(rows - 1);
        }
        self.message = format!("cell page {}/{}", self.cell_page + 1, pages);
        self.message_error = false;
    }

    // ------------------------------------------------------------- keys

    pub fn handle_key(&mut self, key: KeyEvent) {
        if self.mode == Mode::ConfirmZero {
            match key.code {
                KeyCode::Char('y') | KeyCode::Enter => self.confirm_zero(true),
                KeyCode::Char('n') | KeyCode::Esc => self.confirm_zero(false),
                _ => {}
            }
            return;
        }
        if self.mode == Mode::ConfirmQuit {
            match key.code {
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    self.quit_after_save = true;
                    self.save(None);
                    self.mode = Mode::Normal;
                }
                KeyCode::Char('d') => {
                    self.should_quit = true;
                    self.mode = Mode::Normal;
                }
                KeyCode::Char('n') | KeyCode::Esc => {
                    self.mode = Mode::Normal;
                    self.message = "quit cancelled · edits are still in memory".into();
                }
                _ => {}
            }
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if self.mode == Mode::Prompt {
                self.close_prompt();
            } else if self.save_running {
                self.message = "save in progress · wait for it to finish before quitting".into();
            } else if self.history.has_edits() {
                self.mode = Mode::ConfirmQuit;
                self.message = "You have edits in this session. Save before quitting?".into();
                self.message_error = false;
            } else {
                self.should_quit = true;
            }
            return;
        }
        if self.mode == Mode::Prompt {
            self.handle_prompt_key(key);
            return;
        }
        if self.tab == Tab::Help {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q')
            ) {
                self.tab = Tab::Overview;
            }
            return;
        }
        match key.code {
            KeyCode::Char('q') if self.save_running => {
                self.message = "save in progress · wait for it to finish before quitting".into();
            }
            KeyCode::Char('q') if self.history.has_edits() => {
                self.mode = Mode::ConfirmQuit;
                self.message = "You have edits in this session. Save before quitting?".into();
                self.message_error = false;
            }
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('j') | KeyCode::Down if self.tab == Tab::Cells => self.move_cell(1, 0),
            KeyCode::Char('k') | KeyCode::Up if self.tab == Tab::Cells => self.move_cell(-1, 0),
            KeyCode::Char('h') | KeyCode::Left if self.tab == Tab::Cells => self.move_cell(0, -1),
            KeyCode::Char('l') | KeyCode::Right if self.tab == Tab::Cells => self.move_cell(0, 1),
            KeyCode::Char('j') | KeyCode::Down => self.move_cursor(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_cursor(-1),
            KeyCode::Char('g') => {
                self.tree.cursor = 0;
                self.sync_selection();
            }
            KeyCode::Char('G') => {
                self.tree.cursor = self.tree.rows.len().saturating_sub(1);
                self.sync_selection();
            }
            KeyCode::Enter | KeyCode::Char(' ') => self.activate(),
            KeyCode::Char('h') | KeyCode::Left => self.collapse_current(),
            KeyCode::Char('l') | KeyCode::Right => self.expand_current(),
            KeyCode::Char(']') => {
                self.tree.step_tensor(1);
                self.sync_selection();
            }
            KeyCode::Char('[') => {
                self.tree.step_tensor(-1);
                self.sync_selection();
            }
            KeyCode::Char('1') => self.tab = Tab::Overview,
            KeyCode::Char('2') => self.tab = Tab::Heatmap,
            KeyCode::Char('3') => self.tab = Tab::Cells,
            KeyCode::Char('?') => self.tab = Tab::Help,
            KeyCode::Char('+') => self.nudge_slice(Some("+")),
            KeyCode::Char('-') => self.nudge_slice(Some("-")),
            KeyCode::Char('.') => self.nudge_page(Some("+")),
            KeyCode::Char(',') => self.nudge_page(Some("-")),
            KeyCode::Char('/') => self.open_prompt("filter "),
            KeyCode::Char(':') => self.open_prompt(""),
            KeyCode::Char('e') => {
                self.open_prompt(&format!("set {},{} ", self.cell_row, self.cell_col))
            }
            KeyCode::Char('s') => self.open_prompt("scale "),
            KeyCode::Char('n') => self.open_prompt("noise "),
            KeyCode::Char('p') => self.open_prompt("prune "),
            KeyCode::Char('c') => self.open_prompt("clamp "),
            KeyCode::Char('z') => self.request_zero(),
            KeyCode::Char('u') => self.undo(),
            KeyCode::Char('r') => self.redo(),
            KeyCode::Char('S') => self.open_prompt("save "),
            KeyCode::Esc => {
                self.message.clear();
                self.message_error = false;
            }
            _ => {}
        }
    }

    fn handle_prompt_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.close_prompt(),
            KeyCode::Enter => self.submit_prompt(),
            KeyCode::Tab => self.complete(),
            KeyCode::Backspace => self.prompt_backspace(),
            KeyCode::Left => self.prompt_left(),
            KeyCode::Right => self.prompt_right(),
            KeyCode::Home => self.prompt_cursor = 0,
            KeyCode::End => self.prompt_cursor = self.prompt.chars().count(),
            KeyCode::Up => self.prompt_history_move(-1),
            KeyCode::Down => self.prompt_history_move(1),
            KeyCode::Char(c) => self.prompt_push(c),
            _ => {}
        }
    }

    fn move_cursor(&mut self, delta: isize) {
        self.tree.move_by(delta);
        self.sync_selection();
    }

    fn move_cell(&mut self, row_delta: isize, col_delta: isize) {
        let Some(buf) = self.current_buf() else {
            return;
        };
        let (rows, cols) = flatten2d(&buf);
        if rows == 0 || cols == 0 {
            return;
        }
        self.cell_row = (self.cell_row as isize + row_delta).clamp(0, rows as isize - 1) as usize;
        self.cell_col = (self.cell_col as isize + col_delta).clamp(0, cols as isize - 1) as usize;
        self.cell_page = self.cell_row / PAGE;
    }

    fn sync_selection(&mut self) {
        if let Some(row) = self.tree.current() {
            if row.kind == RowKind::Tensor {
                let name = row.name.clone();
                if self.selected.as_deref() != Some(name.as_str()) {
                    self.select(&name);
                }
            }
        }
    }

    fn activate(&mut self) {
        let Some(row) = self.tree.current().cloned() else {
            return;
        };
        if row.kind == RowKind::Group {
            self.tree.toggle_group(&row.group);
            self.rebuild();
        } else {
            self.select(&row.name);
            self.rebuild();
        }
    }

    fn collapse_current(&mut self) {
        let Some(row) = self.tree.current().cloned() else {
            return;
        };
        if row.kind == RowKind::Group {
            if row.expanded {
                self.tree.toggle_group(&row.group);
                self.rebuild();
            }
        } else if let Some(i) = (0..self.tree.cursor)
            .rev()
            .find(|&i| self.tree.rows[i].kind == RowKind::Group)
        {
            self.tree.cursor = i;
        }
    }

    fn expand_current(&mut self) {
        let Some(row) = self.tree.current().cloned() else {
            return;
        };
        if row.kind == RowKind::Group && !row.expanded {
            self.tree.toggle_group(&row.group);
            self.rebuild();
        } else {
            self.tree.move_by(1);
            self.sync_selection();
        }
    }

    // ------------------------------------------------------------- loop

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_quit {
            self.poll_save();
            terminal.draw(|frame| self.render(frame))?;
            if event::poll(std::time::Duration::from_millis(250))? {
                match event::read()? {
                    Event::Key(key) => self.handle_key(key),
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

fn short(name: &str) -> String {
    crate::edit::short(name)
}

fn save_path_argument(line: &str) -> Option<String> {
    let rest = line.strip_prefix("save")?.trim();
    if rest.is_empty() {
        return None;
    }
    let unquoted = rest
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| rest.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(rest);
    Some(unquoted.to_string())
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn common_prefix(items: &[String]) -> Option<String> {
    let first = items.first()?;
    let mut end = first.len();
    for item in items.iter().skip(1) {
        end = end.min(item.len());
        while end > 0 && !item.is_char_boundary(end) {
            end -= 1;
        }
        while end > 0 && first[..end] != item[..end] {
            end -= 1;
            while end > 0 && !first.is_char_boundary(end) {
                end -= 1;
            }
        }
    }
    (end > 0).then(|| first[..end].to_string())
}
