# 嵌套（只进不出） — mathematics voice

$$[a_1,b_1]\supsetneq[a_2,b_2]\supsetneq\cdots\;\Longrightarrow\;\bigcap_{n=1}^{\infty}[a_n,b_n]\neq\varnothing$$

**定义** Cantor 闭区间套：有界闭区间列 I_n=[a_n,b_n] 满足 I_{n+1}⊆I_n，每一层完整含于上一层、层层深入，而交集捕获公共极限点。「只压栈不弹栈」的形式对应是 Dyck 字母表上只含左括号的词：每读一字深度加一、任一前缀深度绝不回落，且因无一括号被闭合，它不是任何 Dyck 词，只是 Dyck 词的真前缀。

**归属** 实分析·Cantor 闭区间套定理（紧致性）；形式语言·Dyck 语言的真前缀（纯压栈词）

**定理** Cantor 闭区间套定理：若有界闭区间满足 [a_{n+1},b_{n+1}]⊆[a_n,b_n] 对一切 n 成立，则 ⋂_{n≥1}[a_n,b_n] 非空；若还有 b_n−a_n→0，则交集恰为单点 {ξ}，ξ=sup_n a_n=inf_n b_n。
