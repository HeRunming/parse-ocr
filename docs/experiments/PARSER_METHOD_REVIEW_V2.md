# 数学书解析方法复盘与 v2 改进

## 结论

`json_comparison_report_unsolved_number_theory.md` 说明，旧流程最主要的问题不是单个 LaTeX
符号，而是结构层的召回与边界定义：它把 A1–F31 的 185 个长文章当成 185 道大习题，同时又
抽取文章内部的一些对象，因而同时产生整节包装重复、续页漏抽、非自足碎片和跨节污染。

因此 v2 不再把“inventory 中的每个目标都抽到了”当成完整性证明。默认流程改为：

1. 结构容器 inventory；
2. 原子数学对象 inventory；
3. 高分辨率直接视觉转录；
4. 独立视觉覆盖审计；
5. 缺失/不完整/冲突/非自足目标的定点重试；
6. 重复、LaTeX、来源与结构一致性验证。

## 对比报告揭示的根因

### 1. 容器与对象混淆

`Exercise A1` 等长条目不是一个可直接形式化的练习，而是包含定义、已知结果、数值例子、
猜想、问题和参考文献的文章容器。将它们与原子对象同时输出造成了报告中的 238 组完整包含型
重复，并虚增 exercise 数量。

### 2. 首问召回不等于整节召回

旧 inventory 能找到所有 A1–F31 标题，但后续提示词只要求大条目保留“明确问题和最少上下文”，
主动舍弃了文章后半段的定理、界、构造和数值例子。A9、B4、F28 等续页因此出现大段漏抽。

### 3. 自证式质量门不可靠

旧质量报告显示 inventory miss、incomplete、conflict 均为 0；这只能证明“抽取结果覆盖了自己
事先列出的 inventory”，不能证明 inventory 覆盖了 PDF。新增对象数量或总字符数也会被整节包装
和重复抬高，不能替代源级覆盖审计。

### 4. 自足性不能靠短文本长度判断

`Are there others?`、`or with m ∤ n.`、`He also calls ...` 等文本即使来自源书，也不能独立作为
Lean/检索输入。修复方式应是扩大可追踪的源跨度或附加上下文，而不是由模型改写数学内容。

## 从 mathdoc-parser 借鉴的机制

已检查 `HeRunming/mathdoc-parser` 当前 `main` 分支。值得吸收的设计包括：

- `semantic_section_id` 与来源顺序；
- source inventory 与最终对象的双向核对；
- span/source provenance；
- 跨节保守去重；
- paragraph/hanging-proof 解析；
- 审计修改的 divergence rollback；
- source proof 与 generated proof 分开标记；
- 严格输出 schema 和阶段缓存。

它当前的长条目补救是把超过 4000 字符的条目重新交给 LLM 拆分，最多保留 8 个子项。这适合
普通过长段落，但不足以稳定处理包含几十个数学对象的 A1/B2 一类文章，而且“保留最长子项”仍
可能漏掉短但重要的命题或例子。它的长文覆盖率还会退化为输出长度/输入长度，容易被重复内容
抬高。因此 v2 采用结构容器隔离和独立视觉覆盖审计，而不是只在末端拆巨型条目。

反过来，我们的直接视觉页级转录、词汇表 gate、高风险字形记录、按目标高分辨率重试和跨页图片
证据，也适合补充 mathdoc-parser 以 OCR Markdown 为主的路线。

## 已落实的 v2 改动

### 结构模型

- inventory 新增 `record_type=section`；
- `structural_label_regex` 描述 A1–F31 一类结构编号；
- `emit_structural_wrappers=false` 时，结构容器不进入环境/QA 数据；
- 最终对象新增 `section_id`、稳定节内 `entry_id`、`source_pages`；
- 结构表单独保存为 `dataset/sections.json`。

### 独立覆盖审计

- 首轮 extraction 后，重新按高分辨率页窗扫描 PDF；
- 审计模型看到“已有对象列表”，只返回可见但未覆盖对象的短 cue 和页界；
- 特别要求检查续页中的已知结果、界、构造、反例、数值例子和公式性结论；
- 审计候选进入 inventory，并由普通 targeted retry 做完整源转录；
- 审计后仍未转录的候选会让最终验证失败，不能静默通过。

### 质检

- 检测结构容器泄漏；
- 检测同节完整包含型重复；
- 检测短指代碎片并触发上下文扩展重试；
- 检测未配对 `$` 与非标准猜想标记；
- 报告没有任何原子对象的结构页；
- 明确把覆盖声明写成 `inventory-plus-independent-audit object recall`，不再宣称绝对完整。

### 来源与证明

- `supporting_text_origin=book_source` 明确书中证明/答案来源；
- 后续模型生成证明不得写回这一字段；
- `context_before/context_after` 预留给不改变 statement 的窄上下文补充。

### Cross-reference 与依赖图

- 把原文导航 `cross_references`、严格逻辑前提 `dependencies` 和非排序语义关联
  `semantic_relations` 分成三个字段；
- 显式引用由规则抽取，隐式关系由文本模型在现有稳定 `entry_id` 候选中选择；
- 每条模型边必须携带源端和目标端的逐字证据，并经过共享术语、重复、自定义、自环、悬空
  和成环检查；
- 允许百科式问题集中的定义后置，再由无环图生成 target-before-source 的
  `formalization_order`；
- 当前 Guy v2 得到 256 条严格依赖、99 条语义关联、74 条显式引用，关系校验为 0 error。

这部分也借鉴了 mathdoc-parser 的“正则显式引用 + 可选 LLM 隐式依赖”两阶段做法，但避免把
普通 `see/cf.` 导航直接等同于 Lean prerequisite，并补上双端证据和独立图审计。

## 针对已知坏页的小规模验证

使用旧缓存，仅对报告点名的 5 个高风险页窗运行独立审计：A1、A9、B4、D16 附近和 F28。

- 5/5 页窗调用成功；
- 返回 78 个旧 inventory 未覆盖的候选；
- A1 找到 Iwaniec、Mignotte 等结果以及后续图形/公式对象；
- A9 找到 Iwaniec、Lou–Yao、Penk、Leech、Herschel Smith、Säfholm 等报告确认的缺项；
- B4 找到大型亲和数对、Lee 规则和 te Riele 例子；
- F28 找到 McIntosh、Wytek、Dănescu–Văjâitu–Zaharescu 等报告确认的缺项；
- 新验证器将这 78 条标为 unresolved，并拒绝把仅完成审计、尚未转录的预览数据判为成功。

这说明新增审计能发现原有“inventory 自己不知道自己漏了”的内容。不过 78 条仍只是候选，必须
经过 targeted transcription、包含去重和视觉抽样后才能进入最终数据。首轮候选中也出现了
图注被当作例子、`or ...`/`who ...` 续句被单独拆出的过细候选；审计提示词已据此增加“图注不
单列、指代续句与必要先行语合并”的约束，最终 gate 仍会把未修复的短碎片列为异常。

## 本轮已完成与后续建议

整书 fresh inventory、coverage audit、targeted transcription、42 条高风险对齐评测、关系抽取和
最终验证均已完成。下一步可在另一本文风不同的教材上回归三类关系的 precision/recall；随后再
同步 skill 仓库，并考虑把结构容器、独立覆盖审计和关系分层设计以 PR 形式反馈给
mathdoc-parser。
