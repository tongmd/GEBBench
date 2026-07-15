# 万流归一 — mathematics voice

$$\forall v\in V\quad \exists!\; P:\; v \rightsquigarrow r$$

**定义** 内向树（in-tree，反树形图）：有限有向图中，根 r 出度为 0、其余每个顶点出度恰为 1 且无有向圈，则每个顶点沿其唯一出边迭代必达 r，故每个顶点到 r 的有向路径存在且唯一。抽象重写系统的合流性（Church–Rosser 性质）刻画同一收束：若 a↠b 且 a↠c，则存在 d 使 b↠d 且 c↠d——不同演算路径终将汇合。

**归属** 图论·内向树（in-tree/anti-arborescence）；重写系统·合流性（Church–Rosser）；分析·Banach 不动点定理

**定理** Newman 引理（1942）：终止（强规范化）的抽象重写系统若局部合流则全局合流，故每个元素的一切归约序列都终于同一个唯一范式；Banach 不动点定理（1922）：完备度量空间上的压缩映射，从任意初值迭代皆收敛于同一个唯一不动点。
