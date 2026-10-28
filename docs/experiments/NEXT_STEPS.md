# 下一步：提取优化的实施路线

基于当前 v3.3（召回 0.84–0.98，精度 0.71–0.84），优化可以分阶段进行。

## 优先级

### 第一阶段：锚点截取（高收益，低风险）

**目标**：消除模型转写错误（Unicode vs LaTeX 符号差异）。

**改动**：
1. 修改 `mine_prompt` 和 `normalize_prompt`，让模型输出 `anchor_start`（6 字）和 `anchor_end`（6 字）而不是整个摘录。
2. 在 `collect_candidates` 里从 PDF 文本层截取两锚点间的内容。
3. 如果没文本层，回退到模型输出。

**预期**：
- must_have 分数从 0.85–0.87 提到 0.92+（符号精准）。
- token 成本下降约 10–15%（输出更短）。
- 去重更精准，duplicate_of_part 可能下降。

**成本**：修改 2 个提示词 + 1 轮 API（约 200k tokens）= 1.5 小时工作。

**验证方法**：
```bash
python3 evaluate.py gold/vershynin.json vershynin/output/dataset/structured_statements.json --pdf pdfs/vershynin.pdf
```
对比 baseline-v3.3 和 v-B-anchors 的 must_have_mean_on_matched。

---

### 第二阶段：粒度控制（中等收益，中等工作量）

**目标**：减少过度原子化导致的重复和上下文缺失。

**改动**：
1. 在 `normalize_prompt` 里加规则：只在明确分隔符或角色转变处拆。
2. 跟踪每条记录是否被拆，及拆分原因。
3. 对 implicit 声明（"可以证明"）检测是否真的在原文有论证。

**预期**：
- 精度从 0.71–0.84 提到 0.75–0.87（虚假拆分减少）。
- atomized_records 数下降 20–30%。
- 总体 F1 分数提高。

**成本**：修改提示词 + 1–2 轮 API + 手工审样本 = 2–3 小时。

**验证方法**：
```bash
python3 evaluate.py ... --out eval/vershynin_v-C-granular.json
```
对比 precision、atomized_records、must_have。

---

### 第三阶段：声明分层（高收益但变数大）

**目标**：按可靠性区分 explicit / implicit / heuristic。

**改动**：
1. 在 mine 和 sweep 里为每条声明标记类型。
2. 跑三个版本：
   - D1：仅 explicit（高精度，低召回）
   - D2：explicit + implicit（平衡）
   - D3：全部（高召回）

**预期**（因书而异）：
- Perelman（导言多）：D3 可能 R=0.85, P=0.78。
- Vershynin（形式化多）：D1 可能 R=0.78, P=0.90。

**成本**：修改提示词 + 3 轮 API = 3–4 小时。

**验证方法**：
```bash
for d in D1 D2 D3; do python3 ablation_runner.py v-$d; done
```
画 R/P/F1 对比表，看每本书的最优点。

---

### 第四阶段：集成优化（E）

**组合**：B（锚点）+ C（粒度）+ D3（全声明）。

**预期**：
- 平衡 recall 和 precision。
- token 成本可控（锚点省下来的与分层的平衡）。

**成本**：已有基础，组合即可 = 1 小时。

---

## 并行运行策略

每个变体可在不同样本上并行跑（样本间独立）。

```bash
# 伪代码
for variant in B C D1 D2 D3 E; do
  for case in vershynin perelman string_topology nesterov; do
    (update_config; run_prose_claims) &
  done
  wait
done
```

每批约 4 个 case × ~300s = 20 分钟，总 6 批 = 2 小时墙钟时间（并行运行）。

---

## 汇总指标

建议用一个 CSV 或 JSON 记录全部变体 × 样本的结果：

```
variant,case,recall_all,recall_unlabeled,precision,f1,tokens,atomized_records,duplicate_of_part,unfaithful_normalized
baseline-v3.3,vershynin,0.88,0.84,0.84,0.86,214284,5,2,0
v-B-anchors,vershynin,0.90,0.86,0.85,0.87,182000,5,1,0
v-C-granular,vershynin,0.87,0.83,0.87,0.85,210000,3,1,0
v-D1-explicit,vershynin,0.78,0.74,0.95,0.84,180000,2,0,0
v-D3-all,vershynin,0.89,0.85,0.82,0.84,220000,6,3,0
v-E-combined,vershynin,0.91,0.87,0.86,0.87,190000,4,1,0
...
```

然后画图：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ablation_results.csv")
for case in df["case"].unique():
    subset = df[df["case"] == case]
    plt.figure()
    plt.scatter(subset["precision"], subset["recall_unlabeled"], s=100)
    for idx, row in subset.iterrows():
        plt.annotate(row["variant"], (row["precision"], row["recall_unlabeled"]))
    plt.xlabel("Precision")
    plt.ylabel("Recall (unlabeled)")
    plt.title(f"{case} - R/P Frontier")
    plt.savefig(f"ablation_{case}_rp.png")
```

---

## 建议实施顺序

1. **立即**：实施 B（锚点截取），验证 must_have 是否真的改善。如果改善 < 2%，就是低价值；如果 > 5%，继续。
2. **如果 B 有效**：立即实施 C（粒度控制），看精度能否从 0.76 提到 0.80+。
3. **平行探索 D**：三个版本各跑一遍，找到每本书的最优 claim 类型组合。
4. **最后集成 E**：把所有有效的改进组合。

---

## 预期最终状态

如果 B、C、D 都有效，可能达到：

| 样本 | baseline-v3.3 | 预期优化后 |
|---|---|---|
| Vershynin | R=0.88, P=0.84 | R=0.91, P=0.86 |
| Perelman | R=0.83, P=0.84 | R=0.86, P=0.85 |
| String Topology | R=0.98, P=0.71 | R=0.98, P=0.75 |
| Nesterov | R=0.94, P=0.52 | R=0.94, P=0.58 |

（数字是保守估计，基于各改进的预期贡献。）

---

## 陷阱检查单

- [ ] 锚点截取是否真的减少了字符级错误？（用 chardet 或 Levenshtein 验证）
- [ ] 粒度控制是否改善了上下文缺失问题？（抽样人工审）
- [ ] 不同声明类型在不同书上的权衡是否可预测？（D1/D2/D3 的结果）
- [ ] 集成时，各改进是否有负交互？（E 不如 B+C+D 的平均）
- [ ] token 成本是否与改进成比例？（不要花 3 倍成本只换 2% recall）
