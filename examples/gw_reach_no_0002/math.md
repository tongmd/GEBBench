# Mathematics voice — gw_reach_no_0002

Node set $V = \{A, B, C, D, E, F\}$, edge set
$$E = \{(A,B), (C,E), (D,A), (D,B), (E,A), (E,C), (E,D), (F,C)\}.$$
Adjacency matrix (rows/columns ordered $A, B, C, D, E, F$):
$$M = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
1 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}.$$

**Question.** Does there exist $k \ge 1$ with $(M^k)_{A,F} > 0$?

**Solution.** No; $(M^k)_{A,F} = 0$ for all $k \ge 1$ (no directed path from $A$ to $F$).
