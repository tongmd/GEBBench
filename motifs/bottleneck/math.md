# 咽喉之点 — mathematics voice

$$uv\notin E:\;\; \min_{\substack{S\subseteq V\setminus\{u,v\}\\ S\ \text{separates}\ u,v}}\lvert S\rvert \;=\; \max\#\{\text{internally disjoint } u\text{–}v \text{ paths}\}$$

**定义** 割点：无向连通图中删去后使连通分量增多的顶点；等价地，v 是割点当且仅当存在两点 x,y≠v，使 x 到 y 的每条路径都经过 v。据 Menger 定理，不相邻两点间取自 V∖{u,v} 的最小点分离集的大小等于其间内部不交路径的最大条数，「一切通路必经同一点」即该值为 1 的情形。

**归属** 图论·割点与 Menger 定理

**定理** Menger 定理（1927）：无向图中不相邻两点 u,v 间内部不交路径的最大条数，等于从 V∖{u,v} 中删去顶点以分离二者所需的最小数目。推论：连通且不相邻的 u,v 之间一切路径必经同一点 w，当且仅当该最大条数为 1。
