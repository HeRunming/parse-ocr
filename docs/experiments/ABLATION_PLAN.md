# 消融实验计划

目标：逐个验证提取粒度、锚点截取、正文覆盖等改进对召回率的实际影响。

## 基准和变体

在同一组 gold（vershynin, perelman, string_topology, nesterov）上运行。每个变体衡量：
- 召回率（R all, R unlabeled）
- 精度（P）
- 去重效果（duplicate_of_part 数）
- token 成本

### A. baseline-direct (已有)
原始 direct_pipeline.py，无 prose_claims。

**预期**：R unlabeled ≈ 0.49–0.59，P ≈ 0.93–1.00。

---

### B. 锚点截取（文本层对齐）

**改进方向**：mine 和 normalize 时，模型只输出 `source_anchor_start` 和 `source_anchor_end`（6-16 字），原文由程序从 PDF 文本层精确截取。

**实现**：
1. 修改 `mine_prompt` 和 `normalize_prompt`，让模型输出两个锚点而不是整个摘录。
2. 在 `collect_candidates` 里，对每条从 PDF 文本层获取两锚点间的内容作为 `source_excerpt`。
3. 如果文本层缺失，回退到模型输出的摘录。
4. 记录回退率和文本层匹配分数。

**预期效果**：
- 消除模型转写错误（比如 Unicode ⌈ vs LaTeX \lceil）。
- 去重更精准，可能减少 duplicate_of_part。
- token 成本略低（输出更短）。

**衡量**：对比 baseline-direct 和 B 的 must_have 分数，字符级相似度。

---

### C. 粒度优化：防止过度原子化

**改进方向**：normalize 时不要无脑拆所有多语句记录。只在以下情况拆：
- 明确分隔符（段落间隔、编号列表、"另一方面"）
- 不同的数学角色（定义 + 定理不拆，但定理 + 证明的结论要拆）

**实现**：
1. 在 `normalize_prompt` 里加更细的拆分规则。
2. 跟踪每条记录有多少部分被原子化。
3. 衡量被拆记录的原子化质量（独立可读性）。

**预期效果**：
- 减少虚假拆分导致的重复和不连贯。
- 可能略降低 recall（有些没拆的原单位会被当成容器），但精度提高。
- normalized_records 数应该下降。

**衡量**：对比 B 和 C 的精度、去重数、atomized_records 数。

---

### D. 正文粒度控制：分层 claim 类型

**改进方向**：mine 和 sweep 里区分三类正文断言：
1. **explicit**: "定义…为…", "称之为…", "我们声称…", "定理 1.3.2 说…"
2. **implicit**: "易见…", "可以证明…", "推论是…"（需推理）
3. **heuristic**: "…暗示…", "物理上…", "这表明…"（直觉、推断）

每类分别输出，验证不同类型对 recall/precision 的贡献。

**实现**：
1. 在 mine 和 sweep 里标记每条 claim 的类型。
2. 后续阶段可选是否包含 implicit 和 heuristic。
3. 跑三个版本：仅 explicit、explicit+implicit、全部。

**预期效果**：
- explicit 应该高精度。
- implicit 会增加 recall，但可能降低精度（一些转写错误）。
- heuristic 权衡可能因书而异（Perelman 多，HDP 少）。

**衡量**：每个版本的 R/P 矩阵。

---

### E. 集成：B + C + D

应用锚点截取 + 粒度控制 + 分层声明。

**预期**：结合各自的优势，达到 R > 0.85, P > 0.80。

---

## 实验流程

对每个变体（B, C, D, E）：

1. **配置调整**：修改 prose_claims.py 和相关 prompt。
2. **运行**：在 4 个样本上各跑一次（mine, normalize, verify, repair, merge）。
3. **评测**：
   ```bash
   python3 evaluate.py gold/$case.json $case/output/dataset/structured_statements.json \
     --pdf pdfs/$case.pdf --out eval/${case}_$variant.json
   ```
4. **汇总**：记录 R/P、token 用量、去重数、unfaithful 率。
5. **可视化**：画表格对比各变体的 R/P/F1 曲线。

---

## 关键指标

每个变体记录：

```json
{
  "variant": "B",
  "case": "vershynin",
  "recall_all": 0.88,
  "recall_unlabeled": 0.84,
  "precision": 0.84,
  "f1": 0.86,
  "tokens": 214284,
  "unfaithful_normalized": 0,
  "duplicate_of_part": 2,
  "atomized_records": 5,
  "normalized_parts": 36,
  "prose_claims_accepted": 21,
  "text_layer_fallback_rate": 0.05,
  "explicit_rate": 0.7,
  "implicit_rate": 0.25,
  "heuristic_rate": 0.05
}
```

---

## 时间估算

- B（锚点截取）：提示词改 + 1 轮 API 调用 = ~3 小时 + 成本
- C（粒度控制）：提示词改 + 手工审几个样本 = ~2 小时
- D（分层声明）：提示词改 + 3 个版本 API = ~6 小时
- E（集成）：组合前面的 = ~2 小时
- 汇总可视化：~1 小时

**总计**：~14 小时工作 + API 成本。可以并行跑不同样本。

---

## 预期发现

- 锚点截取会显著改善 must_have 分数和去重效果。
- 粒度控制会在 implicit 声明上显示权衡（Perelman 或 string_topology 可能受益更多）。
- 分层会揭示不同书对 heuristic 类内容的需求差异。
- 集成变体应该是各样本上最强的。

如果某个变体没有改善，或者成本太高，就当成一条"陷阱"记下来，供下次参考。
