"""GEB benchmark — program voice. sample: rg_btree_0004 (recursive_garden / binary_tree)"""

from dataclasses import dataclass

DEPTH = 4
EXPECTED_LEAVES = 16


@dataclass
class Node:
    left: "Node | None" = None
    right: "Node | None" = None


def build_tree(depth: int) -> Node:
    if depth == 0:
        return Node()
    return Node(left=build_tree(depth - 1), right=build_tree(depth - 1))


def count_leaves(node: Node) -> int:
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


if __name__ == "__main__":
    result = count_leaves(build_tree(DEPTH))
    assert result == EXPECTED_LEAVES, (result, EXPECTED_LEAVES)
    print(f"leaf_count(depth={DEPTH}) = {result}  [matches schema]")
