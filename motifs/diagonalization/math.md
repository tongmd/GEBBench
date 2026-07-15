# 对角线遁逸 — mathematics voice

$$d(n) = 1 - f(n)(n) \;\Rightarrow\; \forall n:\ d \neq f(n)$$

**定义** 设 f:\mathbb{N}\to\{0,1\}^{\mathbb{N}} 为任一枚举表(第 n 行是序列 f(n))。定义反对角序列 d 满足 d(n)=1-f(n)(n),则 d 与表中第 n 行保证性地在第 n 位不同,故 d 与每一行都不相等——d 不在表中。任何试图把 \{0,1\}^{\mathbb{N}} 全部列出的枚举,都被其反对角元素系统性逃逸。

**归属** 集合论 / 可计算性理论,Cantor 对角线法 (diagonal argument)

**定理** Cantor 定理(1891 年对角线证明,Über eine elementare Frage der Mannigfaltigkeitslehre):对任意集合 X,不存在 X 到其幂集 P(X) 的满射;特别地取 X=\mathbb{N},得 P(\mathbb{N})\cong\{0,1\}^{\mathbb{N}} 不可数(其势为连续统),这正是实数不可数的对角线证明。
