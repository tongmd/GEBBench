# Mathematics voice — rg_btree_0004

Recursive definition:
$$T_0 = \operatorname{Leaf}, \qquad T_d = \operatorname{Node}(T_{d-1}, T_{d-1}).$$
Leaf count: $L_d = 2^d$. Total node count: $N_d = 2^{d+1} - 1$.
Branch length recurrence with ratio $r = 0.62$: $\ell_{d+1} = r\,\ell_d$.

**Question.** $L_{4} = ?$

**Solution.** $L_{4} = 2^{4} = 16$ (and $N_{4} = 31$).
