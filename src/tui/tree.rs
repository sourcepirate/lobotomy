//! Flattened layer tree: groups with collapse state and a cursor.

use std::collections::HashSet;

use crate::model::{group_by_layer, relative_label, ModelIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Group,
    Tensor,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub kind: RowKind,
    pub group: String,
    pub name: String,
    pub label: String,
    pub count: usize,
    pub dirty: usize,
    pub expanded: bool,
}

pub struct TreeView {
    pub rows: Vec<Row>,
    pub cursor: usize,
    pub top: usize,
    pub collapsed: HashSet<String>,
    pub filter: String,
}

impl Default for TreeView {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeView {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            cursor: 0,
            top: 0,
            collapsed: HashSet::new(),
            filter: String::new(),
        }
    }

    pub fn rebuild(&mut self, index: &ModelIndex, dirty: &dyn Fn(&str) -> bool) {
        let filter = self.filter.to_lowercase();
        let names: Vec<String> = index
            .names()
            .into_iter()
            .filter(|n| filter.is_empty() || n.to_lowercase().contains(&filter))
            .collect();
        let mut rows = Vec::new();
        for (group, members) in group_by_layer(&names) {
            // A filter should reveal matching leaves even if their group was
            // previously collapsed; clearing the filter restores that state.
            let expanded = !self.collapsed.contains(&group) || !filter.is_empty();
            let dirty_count = members.iter().filter(|m| dirty(m)).count();
            rows.push(Row {
                kind: RowKind::Group,
                group: group.clone(),
                name: group.clone(),
                label: group.clone(),
                count: members.len(),
                dirty: dirty_count,
                expanded,
            });
            if expanded {
                for name in members {
                    if let Some(meta) = index.meta(&name) {
                        let is_dirty = dirty(&name);
                        rows.push(Row {
                            kind: RowKind::Tensor,
                            group: group.clone(),
                            label: format!(
                                "{} [{}]",
                                relative_label(&group, &name),
                                meta.shape_str()
                            ),
                            name,
                            count: 0,
                            dirty: usize::from(is_dirty),
                            expanded: true,
                        });
                    }
                }
            }
        }
        self.rows = rows;
        if self.rows.is_empty() {
            self.cursor = 0;
        } else {
            self.cursor = self.cursor.min(self.rows.len() - 1);
        }
    }

    pub fn current(&self) -> Option<&Row> {
        self.rows.get(self.cursor)
    }

    pub fn move_by(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let next = self.cursor as isize + delta;
        self.cursor = next.clamp(0, self.rows.len() as isize - 1) as usize;
    }

    pub fn toggle_group(&mut self, group: &str) {
        if !self.collapsed.remove(group) {
            self.collapsed.insert(group.to_string());
        }
    }

    /// Move to the next/previous tensor row (skipping group headers).
    pub fn step_tensor(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let mut i = self.cursor as isize;
        loop {
            i += delta;
            if i < 0 || i >= self.rows.len() as isize {
                return;
            }
            if self.rows[i as usize].kind == RowKind::Tensor {
                self.cursor = i as usize;
                return;
            }
        }
    }
}
