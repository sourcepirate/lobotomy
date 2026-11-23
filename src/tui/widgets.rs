//! Custom widgets: a truecolor half-block heatmap and a block histogram.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use crate::model::stats::Stats;

/// Blue → magenta → red gradient.
pub fn heat_color(frac: f32) -> Color {
    let t = frac.clamp(0.0, 1.0);
    let (r0, g0, b0) = (30.0, 60.0, 190.0);
    let (r1, g1, b1) = (220.0, 70.0, 60.0);
    let (r, g, b) = if t < 0.5 {
        let k = t * 2.0;
        (
            r0 + (150.0 - r0) * k,
            g0 + (40.0 - g0) * k,
            b0 + (170.0 - b0) * k,
        )
    } else {
        let k = (t - 0.5) * 2.0;
        (
            150.0 + (r1 - 150.0) * k,
            40.0 + (g1 - 40.0) * k,
            170.0 + (b1 - 170.0) * k,
        )
    };
    Color::Rgb(r as u8, g as u8, b as u8)
}

/// Half-block heatmap: `▀` paints the upper sample as foreground and the
/// lower sample as background, doubling vertical resolution.
pub struct HeatmapWidget<'a> {
    pub grid: &'a [Vec<f32>],
    pub lo: f32,
    pub hi: f32,
}

impl Widget for HeatmapWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.grid.is_empty() {
            return;
        }
        let span = (self.hi - self.lo).abs().max(1e-12);
        let rows = self.grid.len();
        let cols = self.grid.iter().map(|r| r.len()).max().unwrap_or(0);
        if cols == 0 {
            return;
        }
        for y in 0..area.height as usize {
            let top_idx = y * 2;
            let bot_idx = top_idx + 1;
            for x in 0..area.width as usize {
                let c = x.min(cols - 1);
                let top = self
                    .grid
                    .get(top_idx.min(rows - 1))
                    .and_then(|r| r.get(c))
                    .copied()
                    .unwrap_or(self.lo);
                let bot = self
                    .grid
                    .get(bot_idx.min(rows - 1))
                    .and_then(|r| r.get(c))
                    .copied()
                    .unwrap_or(self.lo);
                let tf = (top - self.lo) / span;
                let bf = (bot - self.lo) / span;
                let cell = &mut buf[(area.x + x as u16, area.y + y as u16)];
                cell.set_symbol("▀");
                cell.set_fg(heat_color(tf));
                cell.set_bg(heat_color(bf));
            }
        }
    }
}

/// Compact histogram: one block bar per bin plus an axis label.
pub struct HistogramWidget<'a> {
    pub stats: &'a Stats,
    pub bar_width: usize,
}

impl Widget for HistogramWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let peak = self.stats.hist.iter().copied().max().unwrap_or(0).max(1) as f64;
        let width = self.bar_width.min(area.width as usize);
        for (i, count) in self.stats.hist.iter().enumerate() {
            if i >= area.height as usize {
                break;
            }
            let frac = *count as f64 / peak;
            let filled = (frac * width as f64).round() as usize;
            let mut line = String::new();
            for x in 0..width {
                let c = if x < filled { BLOCKS[8] } else { ' ' };
                line.push(c);
            }
            let edge = self.stats.edges.get(i).copied().unwrap_or(0.0);
            let text = Line::from(vec![
                Span::styled(
                    format!("{edge:>9.3} "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(line, Style::default().fg(heat_color(i as f32 / 19.0))),
                Span::raw(format!(" {count:>7}")),
            ]);
            buf.set_line(area.x, area.y + i as u16, &text, area.width);
        }
    }
}

/// Small segmented tab bar drawn manually so the active tab is obvious.
pub struct TabBar<'a> {
    pub tabs: &'a [&'a str],
    pub active: usize,
}

impl Widget for TabBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut x = area.x;
        for (i, tab) in self.tabs.iter().enumerate() {
            let label = format!(" {tab} ");
            let width = label.chars().count() as u16;
            if x + width > area.x + area.width {
                break;
            }
            let style = if i == self.active {
                Style::default()
                    .bg(Color::Rgb(40, 80, 120))
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            let span = Span::styled(label, style);
            buf.set_line(x, area.y, &Line::from(span), width);
            x += width;
        }
    }
}
