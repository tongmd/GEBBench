"""GEB benchmark — program voice. sample: gw_reach_yes_0001 (graph_world / reachability)"""

from collections import deque

GRAPH = {
    "A": ['B'],
    "B": ['C', 'D'],
    "C": ['A', 'F'],
    "D": ['F'],
    "E": ['F'],
    "F": ['D'],
}

SOURCE, TARGET = 'A', 'F'
EXPECTED = True


def reachable(graph, start, target):
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nxt in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


if __name__ == "__main__":
    result = reachable(GRAPH, SOURCE, TARGET)
    assert result == EXPECTED, (result, EXPECTED)
    print(f"reachable({SOURCE} -> {TARGET}) = {result}  [matches schema]")
