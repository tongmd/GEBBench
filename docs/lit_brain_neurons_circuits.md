# 文献备忘录：神经元定位 / 编辑与向量 / 回路 / 脑对齐 × GEB-Bench

日期：2026-07-18。14 篇 + 1 篇 Nature。经 arXiv/OpenReview/NeurIPS 核实（✓）；少数基于已确认题录+摘要（◐）。

## 1. 总表

| 论文 | 层级 | 核心方法 | 关键发现 | 与我们最相关的一条 |
|---|---|---|---|---|
| Brain-Inspired Functional Networks & Key Neurons (arXiv 2502.20408, 2025) ✓ | neuron→network | fMRI 功能网络(ICA)分析搬到 LLM 神经元激活 | LLM 内存在反复出现的"功能网络"；屏蔽关键网络显著损伤性能 | 方法论镜像：我们用 SAE 特征几何、它用 ICA 网络 |
| Knowledge Neurons (Dai et al., ACL 2022) ✓ | neuron | 积分梯度归因到 FFN 神经元 | 稀疏知识神经元；放大/抑制可调控事实表达 | 定位主义开山；其因果检验正是我们 SAE 主张缺的一环 |
| Skill Neurons (Wang et al., EMNLP 2022) ✓ | neuron | soft-prompt 后按激活预测性筛选 | 技能集中于稀疏神经元、跨任务部分复用 | 预测性筛选可直接用于 motif 神经元 |
| Task-Specific Neurons (~2024) ◐ | neuron | 任务间对比激活 | 任务特异神经元存在、干预选择性损伤 | 四声部=同一任务四种表面——声部×母题双因子设计更强 |
| Language-Specific Neurons (Tang et al., ACL 2024) ✓ | neuron | 跨语言激活对比(LAPE) | 语言特异神经元集中首尾层，**中层更语言无关** | 最同构：语言↔声部、语义↔母题；中层语言无关=我们 SAE 中层痕迹位置 |
| ROME (Meng et al., NeurIPS 2022) ✓ | layer | 因果追踪+MLP 秩一编辑 | 事实存于中层 MLP（主语末 token） | 中层 MLP=键值记忆，是中层 motif 痕迹的机制候选 |
| Does Localization Inform Editing? (Hase et al., NeurIPS 2023) ✓ | 批判 | 追踪效应 vs 编辑成功率 | 二者层级上**不相关** | **最重要警示**：定位≠因果≠可编辑 |
| Function Vectors (Todd et al., ICLR 2024) ✓ | vector | 因果中介筛头、FV 注入 | ICL 任务由中层可搬运向量承载 | **下一个机制实验**：motif vector 跨声部注入 |
| ACDC (Conmy et al., NeurIPS 2023) ✓ | circuit | 贪心边剪枝+激活修补 | 电路极稀疏（IOI: 68/32000 边） | 提取每声部电路再测重叠的工具 |
| Knowledge Circuits (Yao et al., NeurIPS 2024) ✓ | circuit | ACDC 式按知识类型分电路 | 跨关系部分复用 | 跨关系电路重叠=跨声部抽象子电路检验的模板 |
| ModCirc (He et al., ICML 2025, PMLR v267) ✓ | circuit | 任务无关可复用模块电路词表 | 存在跨任务共享模块 | "抽象模块+声部前端"假设的形式化 |
| TopoLM (ICLR 2025, arXiv 2410.11516) ✓ | 架构 | 空间平滑正则训练 | 单元自发形成空间聚簇功能组织 | 功能分区可由简单约束涌现 |
| Kumar et al. (Nat Commun 2024) ◐ | 脑对齐 | 逐注意力头 fMRI 编码 | 头级功能特化与皮层语言网络对应 | FV/电路实验选头的先验 |
| Temporal Structure ↔ Layered Hierarchy (arXiv 2310.07106) ✓ | 脑对齐 | ECoG 时间 vs 层深 | **层深≈脑内加工时间** | 中层痕迹="抽象形成的中期阶段" |
| **Nature: Building compositional tasks with shared neural subspaces** (s41586-025-09805-2) ◐ | 脑 | 灵长类电生理+子空间分析 | 大脑以**共享神经子空间**拼装组合任务 | **与中心问题同构**：共享子空间=跨声部不变表示；拼装=crossvoice；32B"识别不映射"=有积木无拼装 |

## 2. 三个思想脉络

### (a) 定位主义及其危机 → 表述边界
三级阶梯（只能主张第一级）：
1. **表征存在**（已有）："SAE features whose activation is invariant across voices and correlates with motif identity, consistent with—but not demonstrating—a shared mid-layer motif representation"
2. **因果使用**（需干预）：禁写 "computes motifs in layer L"
3. **可编辑**：禁止从特征位置推断干预位置
升级到第 2 级：patch motif 特征子空间，对照 (i) 等范数随机子空间 (ii) 声部子空间 (iii) **非定位层同样 patch**（Hase 教训：效应特异集中于 SAE 层，定位才成立）。

### (b) motif vector 实验（完整设计）
- 提取：母题 X 的 theorem 声部末 token 残差流均值，**减全母题 theorem 总均值**（去声部成分）；进阶按因果间接效应筛头
- 注入：α∈{0.5,1,2,4} 加到 story/scene 末 token，全层扫描；**检验峰值层是否与 SAE 痕迹层重合**
- 测量：P(motif=X) 提升（label logit-diff）
- 对照：错母题向量（应升 P(Z)）、等范数随机（地板）、同声部（天花板）、乱序语境
- 判读：invariant→跨声部达天花板≥50%、中层峰、跨声部向量高余弦；specific→≈随机、按声部聚类。中间结果（部分迁移随深度增强）本身可发表

### (c) 脑对齐证据链
- Nature 子空间论文：PFC 用共享子空间拼装任务——镜像我们两发现：(i) 跨声部共享表示（积木）(ii) **有积木≠会拼装**（32B crossvoice 贴随机）。Related work 首选引用
- Temporal/Layered：层深≈加工时间→中层=中期抽象阶段
- Language Neurons + Kumar + 我们的层扫：首尾层表面特异、中层表面无关三方一致
- TopoLM：功能分区可涌现

## 3. 直接启示
**借用**：Skill Neurons 预测性筛选；Knowledge Circuits 跨条件重叠度量；FV 注入协议；去激活双因子干预（声部×母题）。
**避坑**：Hase 阶梯锁死表述；不写 "located in layer L"；电路重叠比随机子图基线。
**独特位置**：文献全在单一表面/任务内定位；我们检验**跨四表面不变性**+行为/表征双侧几何互证。无一篇同时具备"形式化概念库×多表面实现×双侧几何"。

## 4. Related-work 段落草稿（英文，可直接用）
> A parallel line of interpretability work localizes capabilities to sparse units---factual knowledge (Dai et al., 2022), task skills (Wang et al., 2022), and languages (Tang et al., 2024)---or to transportable objects such as function vectors (Todd et al., 2024) and circuits (Conmy et al., 2023; Yao et al., 2024), though causal localization need not indicate where interventions succeed (Hase et al., 2023). In neuroscience, primate cortex composes novel tasks from shared neural subspaces, and the layered hierarchy of language models mirrors the temporal unfolding of comprehension in the human brain. GEB-Bench contributes the missing test bed for this conversation: a library of formally defined abstract structures, each realized across four surfaces, so that surface-invariant representations---the machine analogue of shared conceptual subspaces---can be probed behaviorally and mechanistically at once.

## 待核对（◐）
Task-Specific Neurons 确切篇目；Kumar 方法细节；Nature 子空间论文实验细节（仅据题录+摘要）
