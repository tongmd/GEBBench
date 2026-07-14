# Mathematics voice — gw_reach_yes_0001

Node set $V = \{A, B, C, D, E, F\}$, edge set
$$E = \{(A,B), (B,C), (B,D), (C,A), (C,F), (D,F), (E,F), (F,D)\}.$$
Adjacency matrix (rows/columns ordered $A, B, C, D, E, F$):
$$M = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}.$$

**Question.** Does there exist $k \ge 1$ with $(M^k)_{A,F} > 0$?

**Solution.** Yes; the minimal such $k$ is $3$, witnessed by
$$A \to B \to C \to F.$$
