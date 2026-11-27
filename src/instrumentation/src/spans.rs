// SPDX-FileCopyrightText: 2026 Ernst D. Larsson
// SPDX-License-Identifier: MIT

//! Span timing, per thread, as a tree: a span opened while another is open on the same thread is its child, so the
//! spans of every code on the thread (a program's stages and phases, a library's walks inside them) nest. Each node
//! holds the total time and the number of calls of its label at its place in the tree.
//!
//! - `span(label)`, coarse and always on: around work that happens a handful of times per call (a whole ERI walk, its
//!   allreduce, an operator build). Two clock reads and a short scan of the open span's children.
//! - `fine_span(label)`, with the `profile` feature only: inside hot loops (per shell-quartet batch, per digest call).
//!   Without the feature it is a zero-sized guard whose drop is empty: no clock read, no branch.
//!
//! Neither belongs inside per-primitive or per-point loops. A caller that runs spans on several threads sums their
//! trees itself (`SpanTree::merge`).

use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

/// Totals per label: label, total time, calls.
pub type Totals = Vec<(&'static str, Duration, u64)>;

struct Node {
    label: &'static str,
    fine: bool,
    children: Vec<usize>,
    total: Duration,
    calls: u64,
}

struct Tree {
    // Node 0 is the root, the thread outside any span; the others in the order first opened.
    nodes: Vec<Node>,
    open: usize,
    // Advanced by `reset`, so that a span open across it records nothing.
    generation: u64,
}

impl Tree {
    const fn new() -> Self {
        Tree { nodes: Vec::new(), open: 0, generation: 0 }
    }

    fn enter(&mut self, label: &'static str, fine: bool) -> (usize, usize) {
        if self.nodes.is_empty() {
            self.nodes.push(Node { label: "", fine: false, children: Vec::new(), total: Duration::ZERO, calls: 0 });
        }
        let parent = self.open;
        // Found by the label's address first, then by its text.
        let children = &self.nodes[parent].children;
        let found = children
            .iter()
            .copied()
            .find(|&c| std::ptr::eq(self.nodes[c].label, label) && self.nodes[c].fine == fine)
            .or_else(|| children.iter().copied().find(|&c| self.nodes[c].label == label && self.nodes[c].fine == fine));
        let node = found.unwrap_or_else(|| {
            let node = self.nodes.len();
            self.nodes.push(Node { label, fine, children: Vec::new(), total: Duration::ZERO, calls: 0 });
            self.nodes[parent].children.push(node);
            node
        });
        self.open = node;
        (node, parent)
    }

    fn leave(&mut self, node: usize, parent: usize, generation: u64, elapsed: Duration) {
        if generation != self.generation || node >= self.nodes.len() {
            return;
        }
        self.nodes[node].total += elapsed;
        self.nodes[node].calls += 1;
        self.open = parent;
    }

    fn totals(&self, fine: bool) -> Totals {
        let mut totals: Totals = Vec::new();
        for node in self.nodes.iter().skip(1).filter(|n| n.fine == fine && n.calls > 0) {
            match totals.iter_mut().find(|(label, _, _)| *label == node.label) {
                Some(entry) => {
                    entry.1 += node.total;
                    entry.2 += node.calls;
                }
                None => totals.push((node.label, node.total, node.calls)),
            }
        }
        totals
    }

    fn snapshot(&self) -> SpanTree {
        let mut out = SpanTree::default();
        if let Some(root) = self.nodes.first() {
            for &child in &root.children {
                self.walk(child, 0, &mut out.nodes);
            }
        }
        out
    }

    fn walk(&self, index: usize, depth: usize, out: &mut Vec<SpanNode>) {
        let node = &self.nodes[index];
        out.push(SpanNode { label: node.label, depth, fine: node.fine, total: node.total, calls: node.calls });
        for &child in &node.children {
            self.walk(child, depth + 1, out);
        }
    }
}

thread_local! {
    static TREE: RefCell<Tree> = const { RefCell::new(Tree::new()) };
}

fn enter(label: &'static str, fine: bool) -> (usize, usize, u64) {
    TREE.with(|tree| {
        let mut tree = tree.borrow_mut();
        let (node, parent) = tree.enter(label, fine);
        (node, parent, tree.generation)
    })
}

fn leave(node: usize, parent: usize, generation: u64, elapsed: Duration) {
    TREE.with(|tree| tree.borrow_mut().leave(node, parent, generation, elapsed));
}

/// A coarse span: open from [`span`] until dropped, its time recorded under its label at its place in this thread's
/// tree.
#[must_use]
pub struct Span {
    node: usize,
    parent: usize,
    generation: u64,
    start: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        leave(self.node, self.parent, self.generation, elapsed);
    }
}

/// Opens a coarse span under the span open on this thread (or at the top).
#[inline]
pub fn span(label: &'static str) -> Span {
    let (node, parent, generation) = enter(label, false);
    Span { node, parent, generation, start: Instant::now() }
}

/// Adds `n` to the calls of `label` under the span open on this thread, with no time: a count of events too many to
/// time one by one (quartets skipped in a walk), recorded once per walk. Nothing for `n = 0`.
pub fn count(label: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    TREE.with(|tree| {
        let mut tree = tree.borrow_mut();
        let (node, parent) = tree.enter(label, false);
        tree.nodes[node].calls += n;
        tree.open = parent;
    });
}

#[cfg(feature = "profile")]
mod fine {
    use std::time::Instant;

    /// A fine span (`profile` builds): as [`super::Span`].
    #[must_use]
    pub struct FineSpan {
        node: usize,
        parent: usize,
        generation: u64,
        start: Instant,
    }

    impl Drop for FineSpan {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            super::leave(self.node, self.parent, self.generation, elapsed);
        }
    }

    /// Opens a fine span under the span open on this thread.
    #[inline]
    pub fn fine_span(label: &'static str) -> FineSpan {
        let (node, parent, generation) = super::enter(label, true);
        FineSpan { node, parent, generation, start: Instant::now() }
    }

    /// Whether fine spans are recorded (the `profile` feature).
    pub const FINE_ENABLED: bool = true;
}

#[cfg(not(feature = "profile"))]
mod fine {
    /// A fine span without the `profile` feature: records nothing.
    #[must_use]
    pub struct FineSpan;

    /// Records nothing without the `profile` feature.
    #[inline(always)]
    pub fn fine_span(_label: &'static str) -> FineSpan {
        FineSpan
    }

    /// Whether fine spans are recorded (the `profile` feature).
    pub const FINE_ENABLED: bool = false;
}

pub use fine::{fine_span, FineSpan, FINE_ENABLED};

/// The coarse spans' totals on this thread, summed over their places in the tree, labels in the order first opened.
pub fn report() -> Totals {
    TREE.with(|tree| tree.borrow().totals(false))
}

/// The fine spans' totals on this thread, as [`report`]; always empty without the `profile` feature.
pub fn fine_report() -> Totals {
    TREE.with(|tree| tree.borrow().totals(true))
}

/// This thread's spans as a tree.
pub fn tree() -> SpanTree {
    TREE.with(|tree| tree.borrow().snapshot())
}

/// Clears this thread's spans. Spans open across it record nothing.
pub fn reset() {
    TREE.with(|tree| {
        let mut tree = tree.borrow_mut();
        tree.nodes.clear();
        tree.open = 0;
        tree.generation += 1;
    });
}

/// One span in a [`SpanTree`]: its label, depth (0 at the top), whether it is fine, its total time (children
/// included) and its number of calls.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanNode {
    pub label: &'static str,
    pub depth: usize,
    pub fine: bool,
    pub total: Duration,
    pub calls: u64,
}

/// Spans as a tree, in depth-first order (each node followed by its children, siblings in the order first opened).
/// `Display` prints it as an indented table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpanTree {
    nodes: Vec<SpanNode>,
}

impl SpanTree {
    /// The nodes, depth-first.
    pub fn nodes(&self) -> &[SpanNode] {
        &self.nodes
    }

}

impl fmt::Display for SpanTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.nodes.iter().map(|n| 2 * n.depth + n.label.len()).max().unwrap_or(0).max(4);
        writeln!(f, "{:<width$}  {:>12}  {:>10}", "span", "seconds", "calls")?;
        for node in &self.nodes {
            let label = format!("{:indent$}{}", "", node.label, indent = 2 * node.depth);
            writeln!(f, "{label:<width$}  {:>12.6}  {:>10}", node.total.as_secs_f64(), node.calls)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(tree: &SpanTree) -> Vec<(&'static str, usize, u64)> {
        tree.nodes().iter().map(|n| (n.label, n.depth, n.calls)).collect()
    }

    #[test]
    fn a_count_adds_calls_without_time_under_the_open_span() {
        reset();
        {
            let _walk = span("walk");
            count("skipped", 5);
            count("skipped", 2);
            count("none", 0);
        }
        assert_eq!(shape(&tree()), vec![("walk", 0, 1), ("skipped", 1, 7)]);
        assert_eq!(tree().nodes()[1].total, Duration::ZERO);
    }

    #[test]
    fn spans_nest_under_the_span_open_on_the_thread() {
        reset();
        {
            let _stage = span("stage");
            for _ in 0..2 {
                let _phase = span("phase");
                let _walk = span("walk");
            }
            let _walk = span("walk");
        }
        let _other = span("other");
        drop(_other);
        assert_eq!(
            shape(&tree()),
            vec![("stage", 0, 1), ("phase", 1, 2), ("walk", 2, 2), ("walk", 1, 1), ("other", 0, 1)]
        );
        let flat: Vec<(&str, u64)> = report().iter().map(|(l, _, c)| (*l, *c)).collect();
        assert_eq!(flat, vec![("stage", 1), ("phase", 2), ("walk", 3), ("other", 1)]);
    }

    #[test]
    fn a_parent_lasts_at_least_as_long_as_its_children() {
        reset();
        {
            let _outer = span("outer");
            let _inner = span("inner");
            std::thread::sleep(Duration::from_millis(2));
        }
        let nodes = tree();
        assert!(nodes.nodes()[0].total >= nodes.nodes()[1].total);
        assert!(nodes.nodes()[1].total >= Duration::from_millis(2));
    }

    #[test]
    fn a_span_open_across_a_reset_records_nothing() {
        reset();
        let open = span("open");
        reset();
        drop(open);
        assert!(tree().nodes().is_empty());
        let _after = span("after");
        drop(_after);
        assert_eq!(shape(&tree()), vec![("after", 0, 1)]);
    }

    #[test]
    fn fine_spans_record_only_with_the_profile_feature() {
        reset();
        {
            let _coarse = span("coarse");
            let _fine = fine_span("fine");
        }
        assert_eq!(fine_report().len(), FINE_ENABLED as usize);
        assert_eq!(tree().nodes().len(), 1 + FINE_ENABLED as usize);
    }

    #[test]
    fn the_table_indents_children() {
        reset();
        {
            let _a = span("a");
            let _b = span("b");
        }
        let text = tree().to_string();
        assert!(text.lines().nth(1).unwrap().starts_with("a "));
        assert!(text.lines().nth(2).unwrap().starts_with("  b "));
    }
}
