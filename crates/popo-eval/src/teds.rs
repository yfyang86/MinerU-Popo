//! Native tree-edit-distance (Zhang-Shasha) and the TEDS title score, replacing
//! the Python `zss` dependency. Costs: insert = delete = 1, rename = 0 when
//! labels are equal else 1 — matching `evaluate.py`.

/// A node in the arena-built tree: a label plus child indices.
struct ArenaNode {
    label: String,
    children: Vec<usize>,
}

/// Build a tree from `(label, level)` nodes under a synthetic `ROOT`
/// (Python `build_tree`): levels are offset by the minimum, and a stack tracks
/// the current ancestry. Returns the arena with the root at index 0.
fn build_arena(nodes: &[(String, i64)]) -> Vec<ArenaNode> {
    let mut arena = vec![ArenaNode {
        label: "ROOT".to_string(),
        children: Vec::new(),
    }];
    if nodes.is_empty() {
        return arena;
    }
    let min_level = nodes.iter().map(|(_, l)| *l).min().unwrap();
    let mut stack: Vec<(usize, i64)> = vec![(0, -1)];
    for (label, raw_level) in nodes {
        let level = raw_level - min_level;
        while stack.last().map(|&(_, l)| l >= level).unwrap_or(false) {
            stack.pop();
        }
        let parent = stack.last().map(|&(i, _)| i).unwrap_or(0);
        let idx = arena.len();
        arena.push(ArenaNode {
            label: label.clone(),
            children: Vec::new(),
        });
        arena[parent].children.push(idx);
        stack.push((idx, level));
    }
    arena
}

/// Nodes excluding the synthetic root (Python `count_tree_nodes`).
pub fn count_nodes_excluding_root(nodes: &[(String, i64)]) -> usize {
    build_arena(nodes).len().saturating_sub(1)
}

/// Postorder representation: labels and leftmost-leaf descendants (1-based).
struct Postorder {
    labels: Vec<String>,
    lld: Vec<usize>,
    keyroots: Vec<usize>,
}

fn postorder(arena: &[ArenaNode]) -> Postorder {
    let mut labels: Vec<String> = Vec::new();
    let mut lld: Vec<usize> = Vec::new();

    // Recursive postorder assigning 1-based indices.
    fn visit(
        arena: &[ArenaNode],
        node: usize,
        labels: &mut Vec<String>,
        lld: &mut Vec<usize>,
    ) -> (usize, usize) {
        let mut first_lld: Option<usize> = None;
        for &child in &arena[node].children {
            let (_, child_lld) = visit(arena, child, labels, lld);
            if first_lld.is_none() {
                first_lld = Some(child_lld);
            }
        }
        labels.push(arena[node].label.clone());
        let idx = labels.len(); // 1-based
        let my_lld = first_lld.unwrap_or(idx);
        lld.push(my_lld);
        (idx, my_lld)
    }
    visit(arena, 0, &mut labels, &mut lld);

    // Keyroots: for each distinct lld value, the node with the largest index.
    let n = labels.len();
    let mut max_index_for_lld: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for i in 1..=n {
        let l = lld[i - 1];
        let entry = max_index_for_lld.entry(l).or_insert(i);
        if i > *entry {
            *entry = i;
        }
    }
    let mut keyroots: Vec<usize> = max_index_for_lld.values().copied().collect();
    keyroots.sort_unstable();

    Postorder {
        labels,
        lld,
        keyroots,
    }
}

/// Zhang-Shasha edit distance between two `(label, level)` node lists, each
/// wrapped under a `ROOT`.
pub fn tree_edit_distance(a: &[(String, i64)], b: &[(String, i64)]) -> usize {
    let pa = postorder(&build_arena(a));
    let pb = postorder(&build_arena(b));
    let n1 = pa.labels.len();
    let n2 = pb.labels.len();

    let mut treedist = vec![vec![0usize; n2 + 1]; n1 + 1];

    for &i in &pa.keyroots {
        for &j in &pb.keyroots {
            let li = pa.lld[i - 1];
            let lj = pb.lld[j - 1];
            // forestdist over the sub-rectangle [li-1..=i] x [lj-1..=j].
            let rows = i - li + 2;
            let cols = j - lj + 2;
            let mut fd = vec![vec![0usize; cols]; rows];
            // offsets so that fd[i1 - (li-1)][j1 - (lj-1)] addresses node (i1, j1).
            for i1 in li..=i {
                fd[i1 - (li - 1)][0] = fd[i1 - 1 - (li - 1)][0] + 1;
            }
            for j1 in lj..=j {
                fd[0][j1 - (lj - 1)] = fd[0][j1 - 1 - (lj - 1)] + 1;
            }
            for i1 in li..=i {
                for j1 in lj..=j {
                    let (oi, oj) = (li - 1, lj - 1);
                    let del = fd[i1 - 1 - oi][j1 - oj] + 1;
                    let ins = fd[i1 - oi][j1 - 1 - oj] + 1;
                    if pa.lld[i1 - 1] == li && pb.lld[j1 - 1] == lj {
                        let rename = if pa.labels[i1 - 1] == pb.labels[j1 - 1] {
                            0
                        } else {
                            1
                        };
                        let val = del.min(ins).min(fd[i1 - 1 - oi][j1 - 1 - oj] + rename);
                        fd[i1 - oi][j1 - oj] = val;
                        treedist[i1][j1] = val;
                    } else {
                        let li1 = pa.lld[i1 - 1];
                        let lj1 = pb.lld[j1 - 1];
                        let prev = fd[li1 - 1 - oi][lj1 - 1 - oj] + treedist[i1][j1];
                        fd[i1 - oi][j1 - oj] = del.min(ins).min(prev);
                    }
                }
            }
        }
    }

    treedist[n1][n2]
}

/// TEDS title score (Python `title_teds_score`): `1 - distance / max_nodes`,
/// clamped to `[0, 1]`.
pub fn title_teds_score(reference: &[(String, i64)], predicted: &[(String, i64)]) -> f64 {
    let distance = tree_edit_distance(reference, predicted);
    let denom = count_nodes_excluding_root(reference)
        .max(count_nodes_excluding_root(predicted))
        .max(1);
    (1.0 - distance as f64 / denom as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_trees_zero_distance() {
        let a = vec![("A".to_string(), 1), ("B".to_string(), 2)];
        assert_eq!(tree_edit_distance(&a, &a), 0);
    }

    #[test]
    fn single_rename_costs_one() {
        let a = vec![("A".to_string(), 1)];
        let b = vec![("X".to_string(), 1)];
        // ROOT==ROOT (0) + rename A->X (1)
        assert_eq!(tree_edit_distance(&a, &b), 1);
    }

    #[test]
    fn insert_node_costs_one() {
        let a = vec![("A".to_string(), 1)];
        let b = vec![("A".to_string(), 1), ("B".to_string(), 2)];
        assert_eq!(tree_edit_distance(&a, &b), 1);
        assert_eq!(count_nodes_excluding_root(&b), 2);
    }
}
