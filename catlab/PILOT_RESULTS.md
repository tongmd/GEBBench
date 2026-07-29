# Functorial-abstraction pilot — OpenRouter 定稿(2026-07-28, 6 模型)

数据:`catlab/results/or_*.json`(全部 OpenRouter,222/222 records、0 空回复、0 unparsed;
空回复一律原 prompt 重试,思维链超长的以 16k token 上限重试)。
分析:`catlab/analysis.json`(`analyze_catlab.py`)、论文槽位:`paper_numbers.py`。
刺激图 = v2 生成器(过 72 图对抗审计;nuisance 孪生结构断言一致、全库无重复 hash)。
yunwu 首轮(3 模型)已归档 `archive_yunwu/catlab_results/`,论文不引用。

## 主结果(catlab Table,论文 tab:catlab)

| 任务(chance) | GPT-5.5 | GPT-5.2 | Opus-4.8 | Gemini-3.1 | Kimi-K2.5 | Qwen72B |
|---|---|---|---|---|---|---|
| endpoint_id(4%) | 80.6% | 75.0% | 58.3% | 73.6% | 70.8% | 56.9% |
| square_thm(25%) | 88.9% | 83.3% | 55.6% | 58.3% | 55.6% | 30.6% |
| **square_story**(25%) | **52.8%** | **36.1%** | **30.6%** | **33.3%** | **44.4%** | **22.2%** |
| id_vs_change 结构臂 | 88.9% | 91.7% | 83.3% | 88.9% | 97.2% | 75.0% |
| id_vs_change nuisance 臂 | 97.2% | 94.4% | 100% | 86.1% | 36.1% | 52.8% |
| composition(n=6) | 67% | 50% | 50% | 67% | 100% | 50% |

## 发现(论文 §sec:catlab 版本)

**F1 · 态射搬运的 voice 梯度成立且更陡。** 定理方格 55.6–88.9%(frontier 五模型;
Qwen72B 30.6%≈chance),故事方格 22.2–52.8%:四/六模型与 25% chance 不可分
(binomial p≥0.09),高于 chance 的两个(GPT-5.5、Kimi)也比自家定理方格低 11–36pp。
配对 McNemar:四个 proprietary frontier 全显著(p≤0.035);Kimi p=0.39、Qwen p=0.58
(它们定理方格本来就低)。

**F2 · 瓶颈是搬运不是看见变化。** frontier 四模型 id_vs_change 结构臂 83–92%、
nuisance 对照 86–100%(误报 0–14%)。开源两家换了另一种失败:Kimi 结构臂 97%
但把 64% 的 cosmetic 孪生也叫成结构变化(过报);Qwen 75%/53%。

**F3 · 双向分离(最有味道的格子)。** cycle→strange_loop:定理方格六模型全对
(36/36 pooled),AFTER 端点命名仅 22/36 —— Opus 和 Qwen72B 0/6 却方格全对
(**搬得动叫不出名**);反向 sym→symbreak:端点命名 69/72 近满,
Gemini 方格 1/6、Kimi 2/6、Qwen 0/6(**叫得出名搬不动**)。K3(退化为识别)不成立。

**F4 · shortcut ceiling。** 定理方格超过自身识别-配对 ceiling 的:Opus +13.9pp、
GPT-5.2 +11.8pp、GPT-5.5 +3.5pp(3/6 存在超出识别的真搬运);
故事方格**六模型全部低于自身 ceiling**(−11 到 −40pp):
变换框架在叙事声部反而有害。

**F5 · identity 陷阱不是故事塌方的原因。** 故事错误选"未变结构的另一故事"
份额 pooled 37/137=27%,低于 1/3 chance(p=0.14);错误流向形式近邻故事,
与主论文 confusion geometry 同型。

**F6 · 复合(n=6,仅方向)。** 模型间差异大(50–100%);
GPT-5.5 的复合错误里仍有 stopped-after-STEP1 形状,样本太小不入正文。

## 与 yunwu 首轮的差异(为什么数字变了)

- OR 的 GPT-5.5/5.2 明显更强(eid 68→81 / 67→75,sq_thm 58→89 / 53→83):
  yunwu `:stable` 代理版本较旧 + 空回复按错计分的偏置。
- 结论方向不变:故事方格塌、定理方格立、双向分离、ceiling 结构同。
- 论文只引用 OR 数字。

## 诚实缺陷(不变)

- n=36/任务/模型(Wilson ±16pp),每边 6 seed;
- 端点可命名性按边异质 → pooled conditional 有 Simpson 混杂,per-edge 是诚实单位
  (论文明写,数据在 analysis.json per_edge_acc);
- 故事选项是现成语料故事(非派生孪生);
- composition n=6 仅方向性。
