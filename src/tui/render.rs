//! Rendering: header, tree, tabbed detail, status and prompt bars.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Cell, Clear, Paragraph, Row, Table, Widget};
use ratatui::Frame;

use crate::edit::short;
use crate::model::stats::{flatten2d, heatmap, sparkline};

use super::tree::RowKind;
use super::widgets::{HeatmapWidget, HistogramWidget, TabBar};
use super::{App, Mode, Tab, CELL_COLS, PAGE};

const ACCENT: Color = Color::Rgb(120, 190, 255);
const ACCENT_BG: Color = Color::Rgb(24, 40, 58);
const DIRTY: Color = Color::Rgb(255, 200, 90);
const ERR: Color = Color::Rgb(255, 120, 120);
const OK: Color = Color::Rgb(140, 230, 160);
const DIM: Color = Color::Rgb(120, 120, 130);

impl App {
    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let rows = Layout::vertical([
            Constraint::Length(1), // header
            Constraint::Fill(1),   // body
            Constraint::Length(1), // status
            Constraint::Length(1), // prompt
        ])
        .split(area);
        self.render_header(frame, rows[0]);
        self.render_body(frame, rows[1]);
        self.render_status(frame, rows[2]);
        self.render_prompt(frame, rows[3]);
        if self.tab == Tab::Help {
            self.render_help_overlay(frame, area);
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let dirty = self.history.dirty_count();
        let spec = &self.index.source_spec;
        let line = Line::from(vec![
            Span::styled(
                " lobotomy ",
                Style::default()
                    .bg(ACCENT_BG)
                    .fg(ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(spec.clone(), Style::default().fg(Color::White)),
            Span::styled(
                format!(
                    "  ·  {}  ·  {} tensors  ·  ",
                    self.index.arch,
                    self.index.tensors.len()
                ),
                Style::default().fg(DIM),
            ),
            Span::styled(
                format!("dirty {dirty}"),
                Style::default()
                    .fg(if dirty > 0 { DIRTY } else { DIM })
                    .add_modifier(if dirty > 0 {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            Span::styled(
                format!(
                    "  ·  {} shown ",
                    self.tree
                        .rows
                        .iter()
                        .filter(|r| r.kind == RowKind::Tensor)
                        .count()
                ),
                Style::default().fg(DIM),
            ),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }

    fn render_body(&mut self, frame: &mut Frame, area: Rect) {
        if area.width < 72 || area.height < 8 {
            let message = if area.width < 72 {
                "Terminal is too narrow for the layer list and tensor view. Resize to at least 72 columns."
            } else {
                "Terminal is too short to show the tensor view. Increase its height."
            };
            frame.render_widget(
                Paragraph::new(message)
                    .style(Style::default().fg(Color::White))
                    .alignment(ratatui::layout::Alignment::Center)
                    .wrap(ratatui::widgets::Wrap { trim: true }),
                area,
            );
            return;
        }
        let tree_width = (area.width / 3)
            .clamp(24, 42)
            .min(area.width.saturating_sub(24));
        let cols =
            Layout::horizontal([Constraint::Length(tree_width), Constraint::Fill(1)]).split(area);
        self.render_tree(frame, cols[0]);
        self.render_detail(frame, cols[1]);
    }

    fn render_tree(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(60, 60, 70)))
            .title(Span::styled(
                format!(
                    " layers ({}) ",
                    self.tree
                        .rows
                        .iter()
                        .filter(|r| r.kind == RowKind::Tensor)
                        .count()
                ),
                Style::default().fg(ACCENT),
            ));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 {
            return;
        }
        // keep the cursor visible
        let visible = inner.height as usize;
        if self.tree.cursor < self.tree.top {
            self.tree.top = self.tree.cursor;
        } else if self.tree.cursor >= self.tree.top + visible {
            self.tree.top = self.tree.cursor + 1 - visible;
        }
        self.tree.top = self.tree.top.min(
            self.tree
                .rows
                .len()
                .saturating_sub(visible.min(self.tree.rows.len())),
        );

        let mut lines: Vec<Line> = Vec::with_capacity(visible);
        for (i, row) in self
            .tree
            .rows
            .iter()
            .enumerate()
            .skip(self.tree.top)
            .take(visible)
        {
            let is_cursor = i == self.tree.cursor;
            let gutter = if is_cursor { "› " } else { "  " };
            let mut spans: Vec<Span> = Vec::new();
            match row.kind {
                RowKind::Group => {
                    let arrow = if row.expanded { "▾" } else { "▸" };
                    let mut style = Style::default()
                        .fg(Color::Rgb(140, 200, 235))
                        .add_modifier(Modifier::BOLD);
                    if is_cursor {
                        style = style.bg(ACCENT_BG);
                    }
                    spans.push(Span::styled(gutter, Style::default().fg(ACCENT)));
                    let room = inner.width.saturating_sub(8) as usize;
                    spans.push(Span::styled(
                        truncate(&format!("{arrow} {} ({})", row.label, row.count), room),
                        style,
                    ));
                    if row.dirty > 0 {
                        spans.push(Span::styled(
                            format!("  *{}", row.dirty),
                            Style::default().fg(DIRTY).add_modifier(Modifier::BOLD),
                        ));
                    }
                }
                RowKind::Tensor => {
                    let selected = self.selected.as_deref() == Some(row.name.as_str());
                    let mut style = Style::default();
                    style = if row.dirty > 0 {
                        style.fg(DIRTY)
                    } else {
                        style.fg(Color::Rgb(210, 210, 220))
                    };
                    if selected {
                        style = style
                            .bg(ACCENT_BG)
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD);
                    } else if is_cursor {
                        style = style.add_modifier(Modifier::UNDERLINED);
                    }
                    spans.push(Span::styled(gutter, Style::default().fg(ACCENT)));
                    spans.push(Span::styled("  ", Style::default()));
                    let room = inner.width.saturating_sub(4) as usize;
                    spans.push(Span::styled(truncate(&row.label, room), style));
                }
            }
            lines.push(Line::from(spans));
        }
        if lines.is_empty() {
            let message = if self.tree.filter.is_empty() {
                "No tensors are available."
            } else {
                "No matches. Press / and Enter to clear the filter."
            };
            frame.render_widget(
                Paragraph::new(message).style(Style::default().fg(Color::Rgb(190, 195, 205))),
                inner,
            );
            return;
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    fn detail_block(&self, title: String) -> Block<'static> {
        Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(60, 60, 70)))
            .title(Span::styled(title, Style::default().fg(ACCENT)))
    }

    fn render_detail(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).split(area);
        let tabs = Tab::names();
        TabBar {
            tabs: &tabs,
            active: self.tab.index(),
        }
        .render(chunks[0], frame.buffer_mut());
        match self.tab {
            Tab::Overview => self.render_overview(frame, chunks[1]),
            Tab::Heatmap => self.render_heatmap(frame, chunks[1]),
            Tab::Cells => self.render_cells(frame, chunks[1]),
            Tab::Help => self.render_help(frame, chunks[1]),
        }
    }

    fn render_overview(&mut self, frame: &mut Frame, area: Rect) {
        let Some(name) = self.selected.clone() else {
            let message = if self.tree.filter.is_empty() {
                "No tensor is selected. Move through the layer list to select one."
            } else {
                "No tensors match this filter. Press /, clear the text, and press Enter."
            };
            let block = self.detail_block(" tensor details ".into());
            frame.render_widget(Paragraph::new(message).block(block), area);
            return;
        };
        let Some(meta) = self.current_meta().cloned() else {
            return;
        };
        let Some(stats) = self.stats() else {
            let read_only = !meta.dtype.is_editable();
            let block = self.detail_block(format!(
                " tensor · {name} · {} ",
                if read_only {
                    "read-only"
                } else {
                    "values unavailable"
                }
            ));
            let mut info = vec![Line::from(vec![
                Span::styled("shape ", Style::default().fg(DIM)),
                Span::styled(meta.shape_str(), Style::default().fg(Color::White)),
                Span::styled("   dtype ", Style::default().fg(DIM)),
                Span::styled(meta.dtype.as_str(), Style::default().fg(ERR)),
                Span::styled("   numel ", Style::default().fg(DIM)),
                Span::styled(meta.numel().to_string(), Style::default().fg(Color::White)),
            ])];
            if read_only {
                info.push(Line::from(
                    "Value statistics and plots are unavailable for this dtype.",
                ));
                info.push(Line::from(
                    "Tensor metadata is still available; editing is disabled.",
                ));
            } else {
                info.push(Line::from(
                    "Tensor values could not be loaded. See the status message for details.",
                ));
            }
            frame.render_widget(
                Paragraph::new(Text::from(info))
                    .block(block)
                    .wrap(ratatui::widgets::Wrap { trim: true }),
                area,
            );
            return;
        };
        let dirty = self.history.is_dirty(&name);
        let block = self.detail_block(format!(
            " tensor · {name}{} ",
            if dirty { " *" } else { "" }
        ));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let chunks = Layout::vertical([
            Constraint::Length(5),
            Constraint::Length(2),
            Constraint::Fill(1),
        ])
        .split(inner);

        let editable = meta.dtype.is_editable();
        let mut info = vec![
            Line::from(vec![
                Span::styled("shape ", Style::default().fg(DIM)),
                Span::styled(meta.shape_str(), Style::default().fg(Color::White)),
                Span::styled("   dtype ", Style::default().fg(DIM)),
                Span::styled(
                    meta.dtype.as_str().to_string(),
                    Style::default().fg(if editable { OK } else { ERR }),
                ),
                Span::styled("   numel ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.3}M", stats.numel as f64 / 1e6),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("min ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.4}", stats.min),
                    Style::default().fg(Color::White),
                ),
                Span::styled("   max ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.4}", stats.max),
                    Style::default().fg(Color::White),
                ),
                Span::styled("   mean ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.4e}", stats.mean),
                    Style::default().fg(Color::White),
                ),
                Span::styled("   std ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.4}", stats.std),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("sparsity ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{:.2}%", stats.sparsity * 100.0),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    if editable {
                        ""
                    } else {
                        "   non-float dtype: read-only"
                    },
                    Style::default().fg(ERR),
                ),
            ]),
        ];
        if let Some(last) = self.history.recent(1).first() {
            info.push(Line::from(vec![
                Span::styled("last edit ", Style::default().fg(DIM)),
                Span::styled(last.clone(), Style::default().fg(DIRTY)),
            ]));
        }
        frame.render_widget(Paragraph::new(Text::from(info)), chunks[0]);

        let spark = sparkline(&stats.hist, chunks[1].width as usize);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("dist ", Style::default().fg(DIM)),
                Span::styled(spark, Style::default().fg(ACCENT)),
            ])),
            chunks[1],
        );

        HistogramWidget {
            stats: &stats,
            bar_width: (chunks[2].width as usize).saturating_sub(24).clamp(8, 60),
        }
        .render(chunks[2], frame.buffer_mut());
    }

    fn render_heatmap(&mut self, frame: &mut Frame, area: Rect) {
        let Some(buf) = self.current_buf() else {
            let message = self
                .current_meta()
                .map(|meta| {
                    format!(
                        "Values for {} ({}) are metadata-only and cannot be plotted.",
                        meta.name,
                        meta.dtype.as_str()
                    )
                })
                .unwrap_or_else(|| "Select a tensor to display its heatmap.".into());
            let block = self.detail_block(" heatmap ".into());
            frame.render_widget(
                Paragraph::new(message)
                    .block(block)
                    .wrap(ratatui::widgets::Wrap { trim: true }),
                area,
            );
            return;
        };
        let inner_width = area.width.saturating_sub(2) as usize;
        let inner_height = area.height.saturating_sub(2) as usize;
        let cols = inner_width.max(8);
        let rows = (inner_height * 2).max(2);
        let hm = heatmap(&buf, rows, cols, self.slice_head);
        let slice = match &hm.slice {
            Some(coords) => format!(
                "slice {:?} ({}/{})",
                coords,
                self.slice_head % hm.total_slices + 1,
                hm.total_slices
            ),
            None => "2D view".to_string(),
        };
        let block = self
            .detail_block(format!(" heatmap · {} · {slice} ", short(&buf.name)))
            .title_bottom(Line::from(vec![
                Span::styled(
                    format!(" {:.3} ", hm.lo),
                    Style::default().fg(Color::Rgb(90, 130, 240)),
                ),
                Span::styled(
                    "▓▓▓▒▒▒░░░▁▁▁",
                    Style::default().fg(Color::Rgb(170, 90, 140)),
                ),
                Span::styled(
                    format!(" {:.3} · +/- pages ", hm.hi),
                    Style::default().fg(Color::Rgb(240, 110, 90)),
                ),
            ]));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let hm = heatmap(
            &buf,
            inner.height as usize * 2,
            inner.width as usize,
            self.slice_head,
        );
        HeatmapWidget {
            grid: &hm.grid,
            lo: hm.lo,
            hi: hm.hi,
        }
        .render(inner, frame.buffer_mut());
    }

    fn render_cells(&mut self, frame: &mut Frame, area: Rect) {
        let Some(buf) = self.current_buf() else {
            let message = self
                .current_meta()
                .map(|meta| {
                    format!(
                        "Values for {} ({}) are metadata-only and cannot be shown or edited.",
                        meta.name,
                        meta.dtype.as_str()
                    )
                })
                .unwrap_or_else(|| "Select a tensor to inspect its values.".into());
            let block = self.detail_block(" cells ".into());
            frame.render_widget(
                Paragraph::new(message)
                    .block(block)
                    .wrap(ratatui::widgets::Wrap { trim: true }),
                area,
            );
            return;
        };
        let (total_rows, total_cols) = flatten2d(&buf);
        if total_rows == 0 || total_cols == 0 {
            return;
        }
        let pages = total_rows.div_ceil(PAGE).max(1);
        self.cell_page = self.cell_page.min(pages - 1);
        self.cell_row = self.cell_row.min(total_rows - 1);
        self.cell_col = self.cell_col.min(total_cols - 1);
        let start = self.cell_page * PAGE;
        let available_cols = inner_columns(area.width).clamp(1, CELL_COLS);
        let col_start = (self.cell_col / available_cols) * available_cols;
        let cols = available_cols.min(total_cols.saturating_sub(col_start));
        let block = self
            .detail_block(format!(
                " cells · {} · flat rows {start}–{}/{} · page {}/{} ",
                short(&buf.name),
                (start + PAGE).min(total_rows),
                total_rows,
                self.cell_page + 1,
                pages
            ))
            .title_bottom(Span::styled(
                format!(
                    " {} · arrows move · e edits · ,/. rows ",
                    cell_coordinate(&buf.shape, self.cell_row, self.cell_col)
                ),
                Style::default().fg(DIM),
            ));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let mut widths = vec![Constraint::Length(7)];
        widths.extend(std::iter::repeat_n(Constraint::Length(11), cols));
        let header: Vec<Cell> = std::iter::once(Cell::from("row"))
            .chain((col_start..col_start + cols).map(|c| Cell::from(format!("c{c}"))))
            .collect();
        let mut rows: Vec<Row> = Vec::new();
        for (r, y) in (start..total_rows).take(PAGE).enumerate() {
            let mut cells: Vec<Cell> = Vec::with_capacity(cols + 1);
            cells.push(Cell::from(format!("{y}")));
            let base = y * total_cols;
            for c in col_start..col_start + cols {
                let v = buf.data.get(base + c).copied().unwrap_or(0.0);
                let style = if v == 0.0 {
                    Style::default().fg(DIM)
                } else if v > 0.0 {
                    Style::default().fg(Color::Rgb(150, 200, 255))
                } else {
                    Style::default().fg(Color::Rgb(255, 160, 160))
                };
                let style = if y == self.cell_row && c == self.cell_col {
                    style
                        .bg(ACCENT_BG)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    style
                };
                cells.push(Cell::from(format!("{v:.4}")).style(style));
            }
            let row_style = if r % 2 == 0 {
                Style::default()
            } else {
                Style::default().bg(Color::Rgb(18, 18, 22))
            };
            rows.push(Row::new(cells).style(row_style));
        }
        let table = Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(DIM).add_modifier(Modifier::BOLD)))
            .column_spacing(1);
        frame.render_widget(table, inner);
    }

    fn render_help(&self, frame: &mut Frame, area: Rect) {
        let block = self.detail_block(" help ".into());
        frame.render_widget(
            Paragraph::new(help_text())
                .block(block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            area,
        );
    }

    fn render_help_overlay(&self, frame: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(8).min(86);
        let height = area.height.saturating_sub(4).min(26);
        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;
        let popup = Rect::new(x, y, width, height);
        frame.render_widget(Clear, popup);
        let block = Block::bordered()
            .border_type(BorderType::Double)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " lobotomy · keys ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ))
            .title_bottom(Span::styled(" Esc / ? closes ", Style::default().fg(DIM)));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);
        frame.render_widget(
            Paragraph::new(help_text()).wrap(ratatui::widgets::Wrap { trim: false }),
            inner,
        );
    }

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        let (mode_label, mode_style) = if self.save_running {
            (
                " SAVING ",
                Style::default()
                    .bg(Color::Rgb(30, 45, 75))
                    .fg(ACCENT)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            match self.mode {
                Mode::Normal => (
                    " NORMAL ",
                    Style::default()
                        .bg(Color::Rgb(30, 60, 45))
                        .fg(OK)
                        .add_modifier(Modifier::BOLD),
                ),
                Mode::Prompt => (
                    " COMMAND ",
                    Style::default()
                        .bg(Color::Rgb(70, 55, 25))
                        .fg(DIRTY)
                        .add_modifier(Modifier::BOLD),
                ),
                Mode::ConfirmZero => (
                    " CONFIRM ",
                    Style::default()
                        .bg(Color::Rgb(80, 50, 20))
                        .fg(DIRTY)
                        .add_modifier(Modifier::BOLD),
                ),
                Mode::ConfirmQuit => (
                    " UNSAVED ",
                    Style::default()
                        .bg(Color::Rgb(80, 35, 35))
                        .fg(ERR)
                        .add_modifier(Modifier::BOLD),
                ),
            }
        };
        let sel = self
            .selected
            .as_deref()
            .map(short)
            .unwrap_or_else(|| "(none)".to_string());
        let msg_style = if self.message_error {
            Style::default().fg(ERR)
        } else {
            Style::default().fg(Color::Rgb(200, 210, 220))
        };
        let line = Line::from(vec![
            Span::styled(mode_label, mode_style),
            Span::styled(format!(" {} ", sel), Style::default().fg(Color::White)),
            Span::styled(
                format!(
                    " undo {} redo {} ",
                    self.history.undo_len(),
                    self.history.redo_len()
                ),
                Style::default().fg(DIM),
            ),
            Span::styled("│ ", Style::default().fg(DIM)),
            Span::styled(self.message.clone(), msg_style),
        ]);
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(Color::Rgb(14, 14, 18))),
            area,
        );
    }

    fn render_prompt(&self, frame: &mut Frame, area: Rect) {
        let (label, text) = match self.mode {
            Mode::Prompt => ("lobotomy> ", self.prompt.clone()),
            Mode::ConfirmZero => (
                "confirm> ",
                "Zero this tensor?  y yes · n / Esc cancel".to_string(),
            ),
            Mode::ConfirmQuit => (
                "quit> ",
                "Unsaved edits  s save & quit · d discard · n / Esc keep working".to_string(),
            ),
            Mode::Normal if self.save_running => (
                "saving ",
                "Model copy is being written and verified · editing and quitting are paused"
                    .to_string(),
            ),
            Mode::Normal if self.tab == Tab::Cells => (
                "cells ",
                "arrows move · e edit value · ,/. page rows · [/ ] change tensor · ? help"
                    .to_string(),
            ),
            Mode::Normal => (
                "press ",
                "? help · : command · / find · e edit · s scale · z zero · u undo · S save · q quit"
                    .to_string(),
            ),
        };
        let line = Line::from(vec![
            Span::styled(
                format!(" {label}"),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(text, Style::default().fg(Color::White)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
        if self.mode == Mode::Prompt {
            use unicode_width::UnicodeWidthStr;
            let x = area.x
                + 1
                + label.width() as u16
                + self.prompt[..char_to_byte(&self.prompt, self.prompt_cursor)].width() as u16;
            frame.set_cursor_position((x.min(area.x + area.width.saturating_sub(1)), area.y));
        }
    }
}

fn inner_columns(width: u16) -> usize {
    // row-number column plus spacing and cell widths (11 characters each).
    width.saturating_sub(10) as usize / 12
}

fn cell_coordinate(shape: &[usize], row: usize, col: usize) -> String {
    if shape.len() <= 2 {
        return format!("row {row}, col {col}");
    }
    let leading = &shape[..shape.len() - 1];
    let mut remainder = row;
    let mut coords = vec![0usize; leading.len()];
    for i in (0..leading.len()).rev() {
        let dim = leading[i].max(1);
        coords[i] = remainder % dim;
        remainder /= dim;
    }
    format!("{:?}, col {col}", coords)
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

/// Clip a label to `width` display columns, adding an ellipsis.
fn truncate(text: &str, width: usize) -> String {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    if width == 0 || text.width() <= width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = ch.width().unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(ch);
        used += w;
    }
    out.push('…');
    out
}

fn help_text() -> Text<'static> {
    let key = |k: &'static str, d: &'static str| {
        Line::from(vec![
            Span::styled(
                format!("  {k:<12}"),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(d, Style::default().fg(Color::Rgb(210, 210, 220))),
        ])
    };
    let head = |h: &'static str| {
        Line::from(Span::styled(
            h,
            Style::default().fg(DIRTY).add_modifier(Modifier::BOLD),
        ))
    };
    Text::from(vec![
        head("navigate"),
        key(
            "j / k, ↑ / ↓",
            "move through the tensor tree    g / G first / last",
        ),
        key("Space, Enter", "expand/collapse group, select tensor"),
        key("h / l, ← / →", "collapse / expand"),
        key("[ / ]", "previous / next tensor"),
        key("1 2 3 ?", "overview · heatmap · cells · help"),
        key("+ / -", "page N-D heatmap slice"),
        key(", / .", "page the cell grid"),
        head(""),
        head("prompt (Esc cancels, Tab completes)"),
        key("/", "filter tensors"),
        key(":", "open the general command prompt"),
        key("s / n / p / c", "scale · noise · prune · clamp"),
        key(
            "e",
            "edit the selected cell: set <flat-row>,<column> <value>",
        ),
        key("z", "zero the tensor (confirm with y; undo with u)"),
        key("u / r", "undo / redo"),
        key("S", "save a copy + lobotomy-diff.jsonl"),
        key("q, Ctrl-C", "quit; choose save, discard, or keep working"),
        key(
            "cells tab",
            "arrows move the selected cell; columns scroll into view",
        ),
        key("prompt ↑ / ↓", "browse recent commands"),
    ])
}
