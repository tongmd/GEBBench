# 收缩不动点 — mathematics voice

$$d(f(x),f(y))\le q\,d(x,y)\ (q<1)\ \text{ on complete }(X,d)\;\Rightarrow\;\exists!\,x^{*}:\ f(x^{*})=x^{*},\quad x_{n+1}=f(x_n)\to x^{*}$$

**定义** 完备度量空间 (X,d) 上的压缩映射 f(Lipschitz 常数 q<1)恰有一个不动点 x*,且从任意初值出发的皮卡迭代 x_{n+1}=f(x_n) 以几何速率 q^n 收敛于 x*;终点不是预先给定的,而是被规则原样送回自身的那一点——完备性不可省略:(0,1) 上的 x↦x/2 即无不动点。

**归属** 分析学 / 动力系统,Banach 不动点定理与皮卡迭代法

**定理** Banach 不动点定理(S. Banach 1922, Fundamenta Mathematicae 3, 133–181):完备度量空间上的压缩映射存在唯一不动点,皮卡迭代自任意初值收敛于它,且先验误差满足 d(x_n,x*) ≤ q^n/(1−q)·d(x_1,x_0)。
