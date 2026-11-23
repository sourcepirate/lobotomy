# lobotomy

[![CI](https://github.com/sourcepirate/lobotomy/actions/workflows/ci.yml/badge.svg)](https://github.com/sourcepirate/lobotomy/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/sourcepirate/lobotomy?sort=semver)](https://github.com/sourcepirate/lobotomy/releases)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange)](https://github.com/sourcepirate/lobotomy)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

A Rust TUI (ratatui) for exploring and surgically editing the weights of
HuggingFace `safetensors` models — layer by layer, live, with undo/redo and
a copy-plus-diff save. Local-first: a repo id is fetched once into a cache,
everything after that happens on disk.

![lobotomy demo: navigating the tensor tree, flipping through the overview,
heatmap, and cells tabs, opening help, and quitting](docs/demo.gif)

## Build and run

Requires Rust 1.88+. Prebuilt binaries for Linux, macOS (Intel + Apple
Silicon), and Windows are attached to each
[release](https://github.com/sourcepirate/lobotomy/releases) (tags `v*`).

```bash
cargo build --release

# local file or directory
target/release/lobotomy ./my-model
target/release/lobotomy ./model.safetensors

# HF repo id (downloaded once into ~/.cache/lobotomy)
target/release/lobotomy Qwen/Qwen2.5-0.5B
target/release/lobotomy Qwen/Qwen2.5-0.5B --revision main
target/release/lobotomy Qwen/Qwen2.5-0.5B --offline     # cache only
```

Visual check without a terminal:

```bash
cargo run --example preview -- ./my-model heatmap 120 34
```

## Screen

```text
 header   repo · arch · tensor count · dirty count
 layers   collapsible groups: relative names + shapes, dirty markers
 detail   [1] overview   [2] heatmap   [3] cells   [?] help
 status   mode · selection · undo/redo · last action (green/red)
 prompt   one-line command bar with cursor, history and tab completion
```

The app opens on the first tensor in a transformer layer when one exists.
**NORMAL** is for navigation; **COMMAND** is for typing. Confirmation states
protect zeroing and quitting with edits. `Esc` cancels the current prompt or
confirmation.

## Keys

| Key | Action |
| --- | --- |
| `j` / `k`, `↓` / `↑` | move through the tensor tree · `g` / `G` first / last |
| `Space`, `Enter` | expand/collapse a group · select a tensor |
| `h` / `l`, `←` / `→` | collapse / expand; in Cells, move between cells |
| `[` / `]` | previous / next tensor |
| `1` `2` `3` `?` | overview · heatmap · cells · help |
| `+` / `-` | page the N-D heatmap slice |
| `,` / `.` | page the cell grid |
| `/` | filter tensors |
| `:` | open the general command prompt (`select`, `page`, `slice`, etc.) |
| `s` `n` `p` `c` | scale · noise · prune · clamp (prompt for params) |
| `e` | edit the focused cell: `set <flat-row>,<column> <value>` |
| `z` | zero the selected tensor (asks for confirmation; undo remains available) |
| `u` / `r` | undo / redo |
| `S` | save a copy + diff log |
| `q`, `Ctrl-C` | quit; with edits, choose save, discard, or keep working |

In the Cells tab, arrows (or `h/j/k/l`) move the highlighted value. Wide
tensors scroll horizontally to keep the selected column visible. For tensors
with more than two dimensions, rows are flattened across the leading
dimensions; the selected coordinate is shown below the table. Prompt `↑`/`↓`
recalls previously submitted commands.

Edits report before→after mean and sparsity in the status line. The heatmap is
a truecolor half-block gradient (`▀`, two samples per cell) with a min/max
legend; 1-D tensors render as a single strip rather than a tiled block.

## Commands

```text
scale <factor> · noise <sigma> · zero · prune <threshold> · clamp <lo> <hi>
set <flat-row>,<column> <value> · select <full-name|unique-fragment|N> · filter <text>
slice <n|+|-> · page <n|+|-> · undo · redo · save [dir]
```

`Tab` completes command names and tensor names after `filter` or `select`.
Paths containing spaces can be quoted: `save "/path/to/my output"`.
Tensor numbers, slice numbers, and cell pages are one-based; cell row/column
coordinates in `set` remain zero-based.

## Saving

Saving never overwrites the source. It writes a new directory containing:

- a new `model.safetensors` (untouched tensors are copied as raw bytes from
  their mmap, only edited tensors are re-encoded)
- copied `config.json`, `tokenizer*`, `special_tokens_map.json`,
  `generation_config.json`
- `lobotomy-diff.jsonl`, one line per edit operation, for replay/audit

The result is re-opened and verified (key set and shapes) before the save is
reported as successful.

Saving runs in the background and reports byte-level copy progress while the TUI
remains responsive. Editing and quitting are paused until verification finishes.
The app refuses to overwrite an existing model file or edit log; choose a new
output directory instead.

## Tests

```bash
cargo test          # model, dtypes, stats, edits, safe save, and TUI interactions
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

The TUI tests drive the app headlessly through `ratatui::backend::TestBackend`
and synthetic key events, asserting on rendered buffer contents.
