# Mathematics voice — gw_cycle_0003

Node set $V = \{A, B, C, D, E, F\}$, edge set
$$E = \{(B,F), (C,B), (C,E), (D,E), (E,A), (E,C), (E,F), (F,D)\}.$$
Adjacency matrix (rows/columns ordered $A, B, C, D, E, F$):
$$M = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 1 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
1 & 0 & 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}.$$

**Question.** Does there exist $k \ge 1$ with $\operatorname{tr}(M^k) > 0$?

**Solution.** Yes; a witness cycle is $$B \to F \to D \to E \to C \to B.$$
