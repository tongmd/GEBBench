# 递归 — mathematics voice

$$T ::= \bullet \mid \mathrm{Node}(T,T),\qquad C_n=\tfrac{1}{n+1}\binom{2n}{n}$$

**定义** 结构由基例与更小的自身实例按同一文法归纳生成：T ::= 基例 ∣ 构造子(T,T)；良基性保证任何展开在有限步内到达基例并逐层返回。对该结构做深度优先遍历，其进/出事件序列恰为合法 Dyck 括号串，且每个 Dyck 词唯一对应一棵这样的树。

**归属** 组合数学·Catalan 数与 Dyck 词；计算理论·结构归纳与调用栈

**定理** 长度 2n 的 Dyck 词、恰有 n 个内部节点的满二叉树、凸 (n+2) 边形的三角剖分，三族对象个数皆为 Catalan 数 C_n=\binom{2n}{n}/(n+1)（Euler–Segner 计数；见 Stanley《Catalan Numbers》），且两两之间存在显式双射。
