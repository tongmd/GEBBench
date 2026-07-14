"""GEB benchmark — program voice. sample: gw_cycle_0003 (graph_world / cycle detection)"""

GRAPH = {
    "A": [],
    "B": ['F'],
    "C": ['B', 'E'],
    "D": ['E'],
    "E": ['A', 'C', 'F'],
    "F": ['D'],
}

EXPECTED = True


def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def dfs(u):
        color[u] = GRAY
        for v in graph.get(u, ()):
            if color[v] == GRAY or (color[v] == WHITE and dfs(v)):
                return True
        color[u] = BLACK
        return False

    return any(color[v] == WHITE and dfs(v) for v in list(graph))


if __name__ == "__main__":
    result = has_cycle(GRAPH)
    assert result == EXPECTED, (result, EXPECTED)
    print(f"has_cycle = {result}  [matches schema]")
