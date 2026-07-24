# GEB-Bench ICLR 实验计划最终报告

## 1. 三个最值得做的实验

### ① Formal Geometry, Not Pixel Geometry — 形式距离 vs 感知距离的回归竞赛(theory lens)

**一句话 insight**:VLM 在自然图像上的错误由 motif 的形式距离(骨架图编辑距离/家族结构)组织,而非图像的感知距离(DINOv2 embedding)——即使刺激是照片,错误也活在形式空间里。

**为什么 novel**:审计未发现 kill。最近的对手是 arXiv:2512.21329("Your Reasoning Benchmark May Not Test Reasoning",结论相反:~80% 是感知错误,但方法是图示刺激上的人工归因)和 Tong et al. "Eyes Wide Shut?"(CVPR 2024,只用了感知预测器一半)。没有任何工作在 VLM 混淆矩阵上跑过 formal-GED vs perceptual-embedding 的 MRQAP 竞赛,也没有任何刺激集在自然照片上把这两个距离实验性解耦——这是 GEB 生成管线独有的能力。定位为对 2512.21329 的定量裁决,而非新问题的发现。

**Protocol**(按 reframe 修正后):
1. **前置检查**:先验证 PERCEPTUAL 与 FORMAL 两个 25×25 预测矩阵解耦(Mantel r < 0.2),不解耦则竞赛无法进行,立即止损
2. PERCEPTUAL = 每 motif ≥5 个 nuisance 重采样渲染的 DINOv2(+CLIP 鲁棒性)pairwise 距离;FORMAL = 主预测器 binary same-family(预注册),次预测器 skeleton spec 图的 networkx GED
3. **加第三预测器**:motif 名称的 text-embedding 距离,分离"看错结构"与"标签空间先验"
4. 响应矩阵:GPT-5.5/Claude/Gemini/Qwen2.5-VL-32B 场景任务混淆计数,补 nuisance-resample 跑数(本来就在计划内)凑到 n≥400 错误事件
5. 统计:dyadic logistic regression + 10k permutation(不用裸 Mantel,矩阵太稀疏);kill-shot 控制:motif 内 per-item accuracy ~ 到本 motif 感知质心距离,应为平
6. **循环性防御**:文档化 confusable families 在任何渲染存在之前就由 skeleton spec 固定

**预期图表**:(a) pairwise 混淆率 vs 两种距离散点 + partial-r 标注;(b) 每模型 β_formal vs β_perceptual 柱状图——每个模型 formal 柱占优即是 money figure。

**时间/成本**:~2-3 天,几乎零新成本(混淆已有日志,embedding 几分钟 3090)。全计划中 insight/成本比最高,先做。

---

### ② Do VLMs Compare Spontaneously? — 比较促进抽象 + near-analog 签名(cogsci lens)

**一句话 insight**:并排两个同 motif 场景能否解锁单场景无法提取的结构,以及 Gentner & Namy 的"近类比优于远类比"发展心理学签名是否在机器上复现——若无指令时失败、有指令时成功,则 VLM 拥有对齐机器但不自发调用。

**为什么 novel**:全部候选中唯一无 killer 的高分项(3.5/4)。最近 prior:Bongard-RWR+(2025)做过每侧图像数消融但从未操纵配对的表面相似度、也无 ensembling foil;"VLMs are Biased"(2505.23941)显示并排在反事实检测上无效(不同任务,只预支了 uninstructed-null 的意外性);多模态 ICL 高/低相似 exemplar 对比(2605.02378)是检索式而非结构定义的"same"。**near-analog 优势签名、ensembling foil、instructed-vs-spontaneous 三分离全部无人占据。**

**Protocol**(按 reframe):5 条件——单场景 / 同 motif 近对(nuisance 相似)/ 同 motif 远对 / foil 对(表面相似但 motif 属混淆家族)/ 指令化比较。修正 condition-4 的指令-计分混淆(统一指令、单图计分)。25 motifs × 2 item-sets × 5 条件,within-item 设计。模型:GPT-5.5、Claude、Gemini、Qwen2.5-VL-72B/7B、InternVL2.5-8B(near-chance 的 7B 是关键——比较能否像拯救四岁儿童一样拯救它们)。混合 logistic + within-item McNemar;**预注册 near×far 交互的功效**(50 items/cell 偏薄,必要时加倍近/远对)。

**预期图表**:每模型 Δaccuracy(条件−单图)dot-and-CI 梯形图,near/far 双色,叠 Gentner & Namy 儿童效应量参考带;第二面板 Δ vs 基线准确率(比较是否恰好帮助最需要的模型)。

**时间/成本**:~100 张新场景(3 天生成+审核)+ <$200 API,共 ~1 周。

---

### ③ 抽象的心理物理函数 — Skeleton→Scene 参数化 morph 阶梯(generative lens)

**一句话 insight**:抽象崩溃是同一个机制、只是各模型工作点不同——从 7B 到 frontier,心理测量曲线的斜率共享,只有阈值移动("不是能不能抽象,而是能穿透多少表面熵")。

**为什么 novel**:Bongard-RWR+(2508.12026)做过同一抽象概念的 synthetic-vs-real 二点对比(伤而不杀:无参数阶梯、无 psychometric 拟合);VRIQ/2512.21329 已确立"realism hurts"结论,所以**不卖"真实感有害",卖不变律**:slope-equality 检验 + threshold ordering 是无人发表过的分解——sigmoid vs 阶跃函数本身就是机制判别(渐进抽象 vs 二值感知管线故障)。明确定位为用连续、程序验证的旋钮取代 Bongard-RWR+ 的二值对比。

**Protocol**(按 reframe 修补两个洞):25 motifs × 7 级自然化阶梯(L0 骨架→L6 photoreal cluttered)× 3 seeds ≈ 525 图;**在每模型阈值附近的 bracketing 层级加 seeds**(3 seeds 不够 pairwise threshold CI);**放弃"从程序重渲染"验证,改为对 L5-L6 生成图做独立结构检查**(gpt-image 会静默破坏拓扑)。6 模型 25-way 评测;混合效应 logistic 拟合 per-model 心理测量函数,LRT 检验 slope×model 交互,bootstrap 阈值排序;per-motif 阈值与混淆家族结构相关。

**预期图表**:经典心理物理图——6 条 sigmoid(accuracy vs 自然化层级),阈值+CI 标注;inset:阈值 vs 模型规模。

**时间/成本**:~$40 生成 + $150 API + 1 天本地 GPU,共 ~1 周。

**排期**:①(2-3天)→ ②与③并行(各1周,②走 API、③吃生成管线+3090),2.5 周内三个全部落地,留半周写作缓冲。

---

## 2. 第二梯队(时间富余或分析近零成本时做)

**Structure-in-the-Telling vs Told(checker 认证的 told/telling 最小对)**:方法真 novel(无人做过程序验证的叙事形式消融最小对),但寓意已被 ARN/Lewis & Mitchell 占据,且 "Sonnet or Not, Bot?" 显示模型能识别固定形式,预测未必成立。按 reframe 做 method-first + 双消融地板条件,~3 天生成 + $150-300。**额外价值:它同时是 story voice 的 reviewer-facing 效度验证**,即使结果平淡也直接加固 benchmark 主张,建议纳入。

**错误构成 × 规模(Machine Relational Shift + 错误熵合并)**:两个 scaling 候选实为一个分析——"错误构成是与准确率不同的 scaling 可观测量",用设计混淆家族做客观分类 + matched-accuracy bootstrap + 条件化置换 z。Opiełka 2403.09734 已定性宣称,你做首个定量曲线。纯日志分析,零新推理(依赖开源阶梯评测日志,而 Qwen2.5-VL-32B 已在跑)。适合作论文一节,不作 headline。

**SAE voice-invariance 跨 gemma 2b→9b→27b**:9b 已在跑,27b 边际成本一天/档。按 reframe 放弃"表征先于行为涌现"(Chen et al. ICLR 2024 已占),改测 usage gap:每个尺度上残差流携带的 voice-invariant 结构都多于行为利用的量,gap 随规模如何闭合。3 个尺度点统计上薄,定位为 interp 支线证据,与实验①的"错误活在形式空间"互为表里。

**Voice×Motif 网格因子化(residual coupling)**:识别策略(边际剥离残差耦合 + held-out-cell CV + 新 motif 迁移)真 novel,Step 3(8 个新 motif 零样本迁移)是 crown jewel 且顺带回答污染质疑。但审计指出的功效问题致命:每 cell 1-2 items 使"零耦合"null 不可证伪——**除非先做仿真功效校准并铸造 ≥5 items/cell,否则不启动**。若②③顺利提前完成,这是第 4 优先。

## 3. 组合策略:论文的中心主张

不要写成"又一个 benchmark + 排行榜"。三个 top 实验与既有发现共同支撑一个中心命题:

> **抽象失败是有规律的(abstraction failure is lawful)——GEB 的设计几何(先验混淆家族 + 参数化 nuisance + 程序认证标签)首次把"模型在抽象上失败"从一个分数变成一个可测量、可分解、可预测的对象。**

论文结构随之成为一条因果链:

1. **现象**(已有发现 1+3):voice gradient 存在,crossvoice 失败主要是上游抽象失败;
2. **失败在哪个空间**(实验①+已有发现 2):错误不是随机的、不是感知的——它们落在概念的形式邻接图里,且 frontier 模型 59% 选同一个错误 motif。设计几何让这从相关观察升级为回归竞赛的定量裁决,正面回应 2512.21329 的"感知瓶颈"论;
3. **失败如何随表面熵展开**(实验③):同一形式程序沿自然化阶梯连续渲染,崩溃是单一机制、模型特定工作点——斜率共享、阈值移动的不变律;
4. **失败能否被认知操作解除**(实验②):比较-促进-抽象测试区分"没有对齐机器"与"有但不自发调用"——这决定 gap 的可干预性,是审稿人最关心的 so-what;
5. **支线**(SAE usage gap + 错误构成 scaling + told/telling 验证):表征层与行为层的一致证词。

这样 benchmark 本身退居为**测量仪器**,论文的贡献变成"关于机器抽象失败的三条定律",每条都有一个已发表对手可正面交锋(2512.21329、Bongard-RWR+、"VLMs are Biased")。这比 25 motifs × 4 voices 的排行榜强一个量级,且三个实验的 null 分支各自可发表(感知距离获胜→重定向到文本 voices;阶跃函数→二值感知故障;比较无效→与 2505.23941 一致的机器-儿童分离)。

## 4. 不做什么

- **Certified Contamination Immunity**:ConStat + Functional Benchmarks + GSM-Symbolic 三连杀;TOST±3% 在 165 items 下功效不足、8 respondents 拟合不了 2PL。降级为 release engineering:发 3 个 fresh splits + 报 split-to-split SD,一段话解决。
- **Formality Ladder(per-voice 临界规模)**:Mirage 直接指控 N* 构造,3 个视觉尺度点不可识别,跨阶梯比较混淆 voice 与模型家族——按原设计结果不可恢复。文本-only offset-vs-shift 版本若做也只值一张图。
- **Adversarial Co-Evolution 作为 headline**:"frontier 共享盲点"是 Goel et al. + Correlated Errors 两篇 2025 论文的摘要;结构归因部分(tests b/c)可留作未来工作。
- **Abstraction Funnel 独立成篇**:Semantic Hub + FARS(2605.09496)近乎直接重叠;zero-lexical-overlap 压力测试角度有价值但两周内排不进,且实验①以更低成本给出同方向证据。
- **Motif Transplant patching**:"Task Vectors are Cross-Modal"(ICLR 2025)杀掉机制与 headline;family-graded steering 版本值得做但需 ~1 周 hook plumbing,性价比输给 top 3。
- **感知 vs binding 探针**:Campbell(NeurIPS 2024)拥有论点,Linear Separability Ceiling 拥有"第三结局"——只剩仪器级增量,scoop 风险高。
- **开模型预测 frontier 错误身份**:Surrogate Fidelity + CAPA 挤压后近于已知事实的传递组合;SAE-null vs raw-geometry 分离若想要,可在实验①框架内顺带一段。
- **Cross-Mapping Triads**:story cell 被 ARN 预支、child/adult 框架被 KiVA 占——实验②以更干净的范式覆盖了其幸存部分(surface 可被形式折扣的假设可在②的 foil 条件讨论)。
- **Fano/MI 界定**:V-information 框架已是 canonical,且变分下界推不出准确率上界(方向性错误);若需要 stimulus-side 效度论证,一个 frozen-encoder 线性探针小节即可,不立独立主张。
- **Crossvoice MIN 组合律**:Press et al. 的 compositionality gap + AgentCoMa 已预答;耦合模型比较在 n=4 crossvoice items/motif 下无功效,除非未来扩到 n≥12。