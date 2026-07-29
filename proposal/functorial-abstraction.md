# Functorial Abstraction：从 invariance 到 naturality 的评审与 pilot 设计

状态:方向评审 + pilot 规格(2026-07)。上游:GEB-Bench v2(1,156 items,arXiv submit/7857640)。

---

## 0. 一句话判断

**方向是对的,而且是当前论文最自然的续集**——但它有三个必须在 item 设计层面就堵死的
shortcut,不堵死的话整个 "morphism preservation" 的测量会退化回我们已经测过的
recognition。本文档给出堵法、可检验的 kill criteria、和一个 ~650 item 的全可验证 pilot。

核心论点值得原样保留:

> A model knows an abstraction not by naming it, but by predicting how it transforms.
>
> **Object recognition does not imply morphism preservation.**

---

## 1. 为什么这是续集而不是新起点(和 v2 发现的连续性)

把现有 benchmark 用提案的语言重写一遍,连续性是精确的,不是修辞:

| v2 已有发现 | 范畴化重述 | 对新方向的含义 |
|---|---|---|
| crossvoice 是每个模型的地板(mapping tax 21–44 pts) | 模型的 transport τ̂_{v→w} 在 **object 层**就弱 | naturality(морphism 层)预期更弱;新任务是在已知弱信号上加细 |
| confusion geometry:错误 2.2× 集中到 formal 近邻 | 模型把 m 和 g(m) 混淆,g 越小混得越狠 | **confusable family 的边就是 g 的候选集**:GED 矩阵免费给出 morphism 库 |
| Kimi-k2.5 反常:开放 25-way 能认,family4 强制细分塌掉(60.0 < scene 67.6,全场唯一) | 它的 "认出 m" 走的是 label 邻域线索,不是结构 | **Yoneda probe 的直接动机**:用干预行为替代命名,这类模型会被当场分离 |
| comparison null:并排第二实例不帮任何模型 | 模型不能从 x, x′ 的**对**里提取关系 | square completion 用"对"作 morphism 的呈现方式,直接踩在这个缺口上 |
| GemmaScope mean-pooled SAE null | 抽象不在静态 latent 里 | transport operator T_{v→w} 实验是这个 null 的正确救法:找 **可组合的 transport law**,不找 motif neuron |
| geometry 回归的 limitation:formal 与 label-prior 共线,分不开 | 25-way 分类天然带 label 语义通道 | Yoneda probe 是**能把这两个通道切开的仪器**(答案里不出现 motif 名) |

特别地:现有 crossvoice 任务 = naturality square 在 **g = id** 时的特例。提案不是加一个
新任务,是把已有任务嵌进一个更大的交换图族里。这一点写进论文动机会非常干净。

---

## 2. 三个必须堵死的 shortcut(设计层面的死穴)

### S1 · Square completion 的 "认出 m′ 就赢" 捷径

给三角(scene(m), scene(m′), story(m)),四选一补 story(m′):模型只要对每个候选做
story2motif、挑出 motif = m′ 的那个,就**零 transport** 拿分。这是 recognition 的换皮。

**堵法(distractor contract,类比 v2 的 nuisance contract):四个候选必须全部
"看起来都像 m′ 方向的变化",只在'发生了什么变换'上不同。** 标准四件套:

- **正解**:g_w 施加于 x_w(和参考对同一个 g,同一 voice 实现);
- **D1 错基底**:同一个 g 施加于别的 motif m″ 的实现(变化对,但起点不对);
- **D2 错变换**:另一个 g′: m → m‴ 施加于 x_w(起点对,变化不对);
- **D3 nuisance 陷阱**:u(x_w),纯 surface 改写(换 palette/换词汇/换分支数),结构不动。

D3 是四件套里最锋利的一颗:它把 "identity vs structural change"(提案第三类任务)内嵌进
每一道 square 题,模型必须区分 E(u(x))=E(x)(invariance)和 E(g(x))=ρ(g)E(x)
(equivariance)。

**并且把捷径造成求解器**:分析阶段跑一个 recognition-only baseline——用被测模型自己的
motif 预测对每个候选打标、按 "挑 motif=m′" 规则作答。**square 准确率必须显著高于这个
shortcut ceiling 才算测到了 transport**;否则 item 设计失败。这是本方向的第一道
kill criterion,必须预注册。

### S2 · morphism 的呈现方式:单个 artifact 编码不了 "被变换过"

一张 "打破对称的蝴蝶" 图,和一张 "生来不对称" 的图,静态看是同一类东西——**单个 artifact
只携带 m′,不携带 g**。故事可以用叙事把 break 演成事件,但 scene 不行。

**决定:morphism 一律以有序对 (x, g_v(x)) 呈现**(before/after)。skeleton voice 里就是
两个可运行的 program(diff 即 g 的实现);story voice 里是同一篇 tale 的两个 telling
(device checker 分别验证两端);theorem voice 只作端点(见 §3)。这个决定同时解释了
为什么 comparison-null 不矛盾:v2 的 comparison 是**同一 motif 的两个实例**(提取
invariance),这里是**一条边的两端**(提取 morphism)——认知上是不同的任务。

### S3 · Yoneda probe 的 "描述即泄题" 捷径

"绕行一周是否翻转定向?" 这句话本身就是 Möbius 的定义,文本里问等于把答案写在题干。

**堵法:probe 必须被执行,不能被描述。** 干预作用在 artifact 上,问的是**结果**:
给 skeleton 程序 + 一个具体干预(交换两个节点的角色、展开一层、复制并平移),让模型预测
执行后的可观测输出(节点数、是否回到原状、定向翻没翻),答案由**真的跑那个程序**给出。
Hom-signature Y(m) = 这组 exact-match 答案的向量。答案空间里永远不出现 motif 名——
这就是把 formal 通道和 label 通道切开的仪器(直接回应论文 limitation)。

代价:scene voice 的 Yoneda probe 需要生成干预后的图(又要过 screening),story voice
需要可验证的叙事后承。**pilot 只做 skeleton voice 的 Yoneda**(程序执行 100% 机械可验),
scene/story 版留给 phase 2。

---

## 3. 各 voice 能不能实现 g:资产侧约束

| voice | g_v 的实现 | 可验证性 | 约束 |
|---|---|---|---|
| skeleton | program transformation(删一个反射、加一个 twist、闭合一条链) | **完全机械**(两端程序都可运行) | 无;这是 anchor voice |
| story | form device 的手术(crab canon 恰好一处镜像破缺;acrostic 断一环) | device checker 可验证两端 | v2 经验:单 device 已有 4/125 拒稿率,**双端一致性会更贵**;初版每条边只做 2–3 对,先证明可行 |
| theorem | **只作端点**:真定理是"找到的",不是"生成的",两个 motif 之间不存在"半步"的真定理 | 端点各自可验证(沿用 v2 定理) | g 必须选 **两端都在 25-motif 库里** 的边——这自动成立,因为 g 就从 confusable family 边里选 |
| scene | (x, g(x)) 需要成对生成、结构差恰好为 g | dual-probe screen 可复用,但要加 "配对一致性" 检查 | 生成器在 hard motif 上 rejection 率高(ρ=−0.48 那条),**pilot 不做 scene**,phase 2 再上 |

结论:**pilot = skeleton + story + theorem 三 voice,全链路无 LLM-judge**,与提案建议一致。

### g 库的选择标准(6 条边)

从 confusable family 边里选,标准:(i) 独立 GED 小(形式上真的近);(ii) 两端 25 库内、
四 voice 资产齐;(iii) g 有每个 voice 的机械实现;(iv) 语义上是**单步**操作(operad 原语
的一次应用)。建议首发:

1. `symmetry → symmetry_breaking`(删一个反射;story: 镜像叙事恰好一处破缺)
2. `tiling → aperiodic_tiling`(保局部规则,消平移周期)
3. `cycle → strange_loop`(闭合游走 + 层级纠缠;GEB 本尊)
4. `isomorphism → projective_duality`(两系统词典 → 单系统内对合;v2 里最硬的混淆边之一)
5. `nesting → recursion`(裸嵌套 + 统一生成规则)
6. `recursion → contraction_fixed_point`(生成式展开 → 收敛塌缩;v2 第一大 story 混淆
   `contraction→recursion` 19 次,反向施工)

注意 4 和 6 恰好是 frontier 模型的 top 混淆边——**预注册预测**:square 准确率按边分层后
与该边两端的 v2 混淆率负相关。verified 的话,confusion geometry 和 naturality 失败被
同一个几何解释,论文的 "lawful failure" 主线直接延长。

---

## 4. 三类任务的精确定义(pilot 版)

### T1 · Naturality Square Completion(~300 items)

给参考对 (x_v, g_v(x_v)) + 目标 voice 的 x_w,四选一(§2 S1 四件套)选出 g_w(x_w)。
覆盖 3 个 voice-pair × 2 方向(theorem 只作目标端不作参考端时用端点定理呈现)。
评分:exact match,chance 25%。

**主指标不是裸准确率**,是条件量:
P(square 对 | 该模型在 v2 里能认出 m 和 m′ 的两端)。
这一步把 "认不出端点" 和 "搬不动边" 分开——正是标题主张的操作化。

### T2 · Composition Preservation(~150 items)

链 m₀ →g₁ m₁ →g₂ m₂(6 条边里能接上的链,如 nesting→recursion→contraction)。
三种呈现:一步到位对 (x, g₂g₁x)、两步分呈、跨 voice 复合。模型输出闭集里的
(中间 motif, 端 motif) 或复合变换名——**复合在闭集词表上查表定义**,不需要模型输出
真的"态射"。

FVR 必须条件化定义,否则被逐步难度混淆:
**FVR = P( F̂(g₂∘g₁) ≠ F̂(g₂)∘F̂(g₁) | 两个单步各自都答对 )**。

### T3 · Identity vs Structural Change(~200 items)

给 (x, y),四选一:y 是 u(x)(纯 nuisance)/ g₁(x) / g₂(x) / 无关。把 v2 的 nuisance
contract 从"承诺"升级成"被测项"。这也是 equivariance 文献里 categorical ≻ group 的实证
落点:g 多数不可逆(contraction、aperiodic 化),group action 框架根本写不下来——引
categorical DL position paper(arXiv 2402.15332)的理由就在这,不是装饰。

### 规模与统计

650 ± 150 items,全 exact-match。每 item 挂 metadata:边、voice-pair、方向、distractor
型号。分析:κ、逐边分层、McNemar 对 (T1 | endpoints-known) 与 family4 的配对差、
shortcut-ceiling 对照(§2 S1)、FVR。frontier 3 模型 + Qwen 梯队,单轮 API 成本
≈ 650 × 6 模型 ≈ 4k calls,一晚上的量。

### Kill criteria(预注册,做不到就杀/改)

- **K1**:square 准确率 ≤ recognition-only shortcut ceiling(任一 frontier 模型)
  → item 设计失败,回炉 distractor contract;
- **K2**:story voice 的 g_story 双端 device 通过率 < 60%(生成侧)→ story 从 pilot 撤下,
  只保 skeleton+theorem,方向降级为 "skeleton-anchored naturality";
- **K3**:所有模型 T1 条件准确率 ≈ family4 准确率(即 "能细分端点就能补方格",无 transport
  增量)→ 标题主张不成立,如实报告为 "morphism preservation reduces to fine
  discrimination"——这本身也是可发表的负结果,但故事要换。

---

## 5. Yoneda probes(独立小节,可先行)

最小可行版:**skeleton-only,25 motif × 6–8 个干预,全部程序执行验证**,~180 items。
产出 Y(m) 行为签名矩阵后,两个立刻能做的分析:

1. **签名可分性**:Y(m) 的行是否把 25 motif 分开(它们在设计上应该正交);模型的
   签名准确率 vs 25-way 命名准确率的散点——**Kimi-k2.5 预注册预测:命名高、签名低**,
   偏离对角线最远。这一张图值一个 finding。
2. **回填 v2 limitation**:用签名准确率替换 label-prior 做 geometry 回归的仪器变量,
   把 formal/label 两通道的共线性打断。

这个小包甚至可以作为 **当前论文 camera-ready / rebuttal 的加固实验**,不用等新论文。

## 6. Latent transport operators(机制侧,kiti 可跑)

规格:Qwen2.5-VL-32B(本地权重)抽 scene/story/theorem 实现的 layer-l 表征;
最小二乘拟合 T_{v→w}(motif 类均值 + item 级);检验:
(a) 三角一致 T_{b→c}T_{a→b} ≈ T_{a→c},T_{v→v} ≈ I;
(b) G_g^v 取 m′,m 类均值差,ε_nat = ‖T G − G T‖;
(c) **回归:ε_nat 是否预测该模型在同边同 voice-pair 的行为 square 错误率**。
(c) 是 money plot:机制量预测行为量,GemmaScope null 的正确续写(从静态 latent 转向
transport law)。全部在 kiti 3090 上可完成,不新增 API 成本。

## 7. Operad(最大野心,单独一篇的量级)

判断:**方向对、但不是 pilot**。三个原因:

- 复合 motif 的**验证机械化**是新基建:skeleton 免费(程序天然复合),story 的复合 device
  checker 要从头写,scene 撞生成器墙(v2 的 ρ=−0.48 会在复合结构上更糟);
- colored operad 的形式化必须付租金:测的是 **held-out 组合、更深的树、不同括号化**
  (associativity of realization 是真正新颖的 probe),不是术语升级;
- 与 arXiv 2606.13634(operads for question decomposition,纯文本)的分界线清楚:
  他们测推理链的 operadic consistency,我们测**跨模态实现**的 operadic algebra——
  相关不重叠,引用并划界。

路线:naturality 论文(§4–6)先落地,operad 作为其 discussion 里的 "generative
extension",等 6 条边的机械化经验攒够了再开。

## 8. 优先级排序(资源视角)

| 顺位 | 内容 | 依赖 | 成本 | 产出 |
|---|---|---|---|---|
| 1 | T1+T3 squares(skeleton+story+theorem) | g 库 + distractor contract | 生成中等,API 一晚 | 标题主张 |
| 2 | Yoneda skeleton 包 | 无(可立刻做) | 极低 | 反 label-shortcut 仪器,可回填现论文 |
| 3 | T2 / FVR | T1 基建 | 低增量 | composition law |
| 4 | transport operators | 开源权重(kiti 就位) | GPU 数天 | 机制—行为桥 |
| 5 | operad 泛化 | 1–4 的全部机械化经验 | 高 | 下一篇论文 |

---

## 9. 相关工作定位(新论文的 related work 骨架)

- Gentner structure-mapping:类比 = 关系对齐;square completion 是 **二阶类比**
  (transformation 之间的类比),配上 v2 的 comparison-null,正好构成
  "一阶比较不帮忙、二阶映射测不到" 的递进;
- categorical deep learning(2402.15332):非可逆操作要范畴不要群——T3 的理论出处;
- operads for LLM reasoning(2606.13634):operadic consistency 纯文本先例,划界见 §7;
- ARC/Raven/Bongard:变换类任务存在,但 single-voice、合成图;我们的差异化仍是
  **同一 g 的跨 voice 实现**。

## 10. 术语纪律

论文里 "functor" 只作设计语言:E 是分类器不是函子,严格陈述一律落到可操作定义
(闭集词表上的交换性、exact match)。所有范畴论叙述要能被一个不懂 CT 的 reviewer
当作普通实验设计读懂——和 v2 的 nuisance contract 同一纪律。CT 熟练的 reviewer 会盯
"F(g₂∘g₁)=F(g₂)∘F(g₁)" 这类式子的类型正确性,提前把复合定义成查表就没有攻击面。
