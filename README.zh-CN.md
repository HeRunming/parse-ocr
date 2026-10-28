# parse-ocr

[English](README.md) | [中文](README.zh-CN.md)

`parse-ocr` 把数学 PDF(原生电子版或 OCR 后的扫描版)转换为结构化 JSON。它抽取定义、定理、引理、命题、
推论、例子、习题、注记、算法,以及书中印出的与之对应的证明、提示和答案,并导出 QA/VQA 记录。

设计原则是**页面图像是唯一权威**。视觉语言模型(VLM)读取渲染后的页面。PDF 文本层(如果存在)只用于
定位、排序和去重模型读出的内容,绝不替代模型转写的数学内容。

## 目录

1. [输出](#输出)
2. [安装](#安装)
3. [快速开始](#快速开始)
4. [流水线](#流水线)
5. [命令参考](#命令参考)
6. [配置](#配置)
7. [保证与不保证](#保证与不保证)
8. [评测](#评测)
9. [仓库结构](#仓库结构)
10. [开发](#开发)
11. [已知局限](#已知局限)

## 输出

一次运行在 `<output>/dataset/` 下写出两层数据,两层的约定不同。

| 层 | 文件 | 由谁写出 | 约定 |
|---|---|---|---|
| 证据层 | `environments.json`(另有 `.jsonl`、`.md`) | `direct build` | 每个印出的环境一条记录。`statement` 和 `supporting_text` 是 VLM 对页面的转写;有文本层时会定位到文本层上。不做改写。 |
| 证据层(QA 视图) | `qa_pairs.json` | `direct build` | `environments.json` 的扁平视图:`question` = `statement`,`answer` = `supporting_text`,`answer_role` 取 `proof`、`hint`、`solution` 等。 |
| 结构化层 | `structured_statements.json` | `prose-claims merge` | 证据层记录加上从正文中挖掘的命题。每条带有模型写的 `normalized_statement`、原文 `source_statement`、`origin` 和校验器的 verdict。不修改 `environments.json`。 |

每条记录都保留来源信息:`source_pages`、`source_pdf_page`,有文本层时还有 `source_spans`
(`source_span_id`、`pdf_page`、`bbox`)。

**证据层记录**(`environments.json`,节选):

```json
{
  "id": "theorem:2.5", "entry_id": "3.2.004", "section_id": "3.2",
  "kind": "theorem", "label": "2.5", "title": "Optional printed title",
  "statement": "Source-faithful Markdown/LaTeX",
  "supporting_text": "Source proof, derivation, answer, hint, or solution",
  "supporting_text_role": "proof", "supporting_text_origin": "book_source",
  "source_pages": [17, 18], "source_pdf_page": 17,
  "source_spans": [{"source_span_id": "p17.b4-5.t12-49", "pdf_page": 17, "bbox": [18.7, 121.1, 343.3, 196.0]}],
  "uncertain_spans": [], "glossary_context": []
}
```

**结构化层记录**(`structured_statements.json`)额外包含:

| 字段 | 取值 |
|---|---|
| `origin` | `labeled_environment`、`unlabeled_environment`、`labeled_environment_atomized`、`unlabeled_environment_atomized`、`prose_derived` |
| `kind` | `definition`、`notation`、`theorem`、`lemma`、`proposition`、`corollary`、`claim`、`fact`、`conjecture`、`assumption`、`example`、`exercise`(环境记录另有 `remark`、`algorithm`) |
| `assertion_status`(正文命题) | `proved_in_text`、`stated_without_proof`、`cited_result`、`recalled_standard`、`left_to_reader`、`conjectured`、`heuristic`、`definitional` |
| 校验 `verdict` | `faithful`、`minor_issue`、`unfaithful`、`deterministic_verbatim`(未经模型改写)、`unverified` |

`*_atomized` 表示一个印出的环境被拆成了多条原子命题,此时 `uid` 为 `<id>#<n>`。被判为 `unfaithful`
的正文命题写入 `prose_claims_rejected.json`,不进入结构化层。

字段级 schema:[`docs/reference/output-schema.md`](docs/reference/output-schema.md)。

## 安装

需要 Python 3.9–3.13,以及一个 OpenAI 兼容的 VLM 接口(本地或远程均可)。

```bash
pip install -e .
# 开发环境:
pip install -e '.[dev]'
pytest
```

安装后环境中会有 `parse-ocr` 命令。

## 快速开始

```bash
export VLM_BASE_URL=http://127.0.0.1:3000/v1
export VLM_API_KEY=your-api-key

# 1. 生成 my_project/run.json 和 my_project/book_profile.json。
#    加引号的 '${VLM_BASE_URL}' 按字面写入配置,加载配置时再展开。
parse-ocr init book.pdf my_project --vlm-base-url '${VLM_BASE_URL}' --vlm-model gpt-5.5

# 2. 检查 PDF、依赖和凭据。--network 还会探测接口。
parse-ocr preflight --config my_project/run.json --network

# 3. 渲染样例页,让 VLM 起草 profile。
#    审阅 output/inspection/book_profile.suggested.json,把正确的部分抄进 book_profile.json。
parse-ocr inspect --config my_project/run.json --auto

# 4. 运行全部阶段(含校验)。
parse-ocr direct run --config my_project/run.json
```

结果在 `my_project/output/dataset/`。先看 `environments.json`、`qa_pairs.json` 和 `quality_report.json`。

`run.json` 中的相对路径以 `run.json` 所在目录为基准,而不是当前目录。API key 只从 `vlm.api_key_env`
指定的环境变量读取(默认 `VLM_API_KEY`),不会写入 `run.json` 或任何输出文件。

## 流水线

`parse-ocr direct run` 按顺序执行下列阶段。每个阶段读取上一阶段在 `<output>/direct/` 下的 checkpoint,
已缓存的工作会跳过,除非传入 `--force-stage`。

| # | 阶段 | 作用 | 主要输出 |
|---|---|---|---|
| 1 | `glossary` | 读取记号表和符号页 | `direct/glossary/`,之后写入 `dataset/symbol_glossary.json` |
| 2 | `inventory` | 在相互重叠的页窗口中列出每个印出的对象(kind、label、页码范围) | `direct/inventory/index.json` |
| 3 | `extract` | 以抽取缩放比例从页面图像转写每个已登记的对象 | `direct/extraction/windows/` |
| 4 | `retry` | 先运行 `reconcile` 和 `coverage-audit`,再从扩展后的页范围重新抽取缺失的对象、缺失的证明/答案以及检测到的空缺 | `direct/retry/`、`direct/coverage_audit/` |
| 5 | `build` | 合并、去重并在文本层上定位;写出证据层和审计台账 | `dataset/environments.json`、`qa_pairs.json` 等 |
| 6 | `verify` | 对 `dataset/formula_conflicts.json` 中各次转写不一致之处重读高倍裁剪图,然后重新 build | `direct/source_verification/` |
| 7 | `relationships` | 显式交叉引用,加上模型提出的隐式依赖 | `dataset/dependency_graph.json` |
| 8 | `prose-claims` | 仅在 `prose_claims.enabled` 为 true 时运行:mine → normalize → verify → repair → merge | `dataset/structured_statements.json` |
| 9 | `validate` | 对数据集做确定性检查 | `dataset/quality_report.json` |

`reconcile` 和 `coverage-audit` 也可以单独运行。在 `run` 中它们在 `retry` 内部执行。

```
 PDF ──render──► page images ──VLM──► inventory ──► extraction ──► retry (reconcile + coverage audit)
  │                                                                     │
  └──text layer (if any)──► locate / order / dedup ───────────────► build ──► verify ──► relationships
                                                                                            │
                                         validate ◄── prose-claims (optional) ◄─────────────┘
```

### 其他路线

| 路线 | 命令 | 适用场景 |
|---|---|---|
| MinerU | `parse-ocr mineru {mineru,segment,transcribe,build,validate,run}` | 已有 MinerU 导出(本地、缓存或 API)。裁剪图由 VLM 重读。`--ids` 把 `transcribe` 限制在指定对象上。 |
| Hybrid | `parse-ocr hybrid {segment,align,reconcile,build,refine,run}` | direct 路线已稳定,且有独立的 TeX/XML/HTML 导出。第二来源文本用于提高召回和改进边界,绝不覆盖以 PDF 为依据的数学内容。见 [`docs/reference/hybrid-route.md`](docs/reference/hybrid-route.md)。 |

## 命令参考

`parse-ocr <command> [args]` 调用 `parse_ocr.<module>.main()`,与 `python -m parse_ocr.<module>` 等价。
除 `init` 外,所有命令都接受 `--config path/to/run.json`。

| 命令 | 模块 | 参数 | 写出 | 退出码 |
|---|---|---|---|---|
| `init` | `init_project` | `pdf project_dir [--book-id] [--output] [--vlm-base-url] [--vlm-model] [--vlm-key-env] [--route direct\|mineru] [--mineru-backend auto\|cache\|local\|api]` | `run.json`、`book_profile.json` | 0 |
| `preflight` | `preflight` | `[--network]` | `output/preflight.json` | 通过为 0,任一检查失败为 2 |
| `inspect` | `book_inspector` | `[--pages 12] [--zoom 1.6] [--auto]` | `output/inspection/manifest.json`;加 `--auto` 时还有 `profile_suggestion.json`、`book_profile.suggested.json` | 0 |
| `direct` | `direct_pipeline` | `<stage> [--force-stage]` | 见[流水线](#流水线) | 任一阶段失败时非 0 |
| `verify` | `verify_source_conflicts` | `[--force-stage] [--candidate-id ID ...]` | `direct/source_verification/{resolutions,canonical_resolutions,transcription_resolutions}.json` | 缺少 `dataset/formula_conflicts.json` 时非 0 |
| `relationships` | `extract_relationships` | `[--force-stage] [--reuse-existing-graph]` | `dataset/dependency_graph.json` | `relationships.fail_on_request_errors` 为 true(默认)且有请求错误时非 0,并保留原有的图 |
| `prose-claims` | `prose_claims` | `{mine,normalize,verify,repair,merge,run} [--force-stage]` | `dataset/prose_claims*.json`、`structured_statements*.json` | 任一阶段失败时非 0 |
| `validate` | `validate_dataset` | — | `dataset/quality_report.json` | 通过为 0,任一错误级检查失败为 2 |
| `mineru` | `mineru_pipeline` | `<stage> [--force-stage] [--ids ID ...]` | `output/mineru/`、`dataset/` | 任一阶段失败时非 0 |
| `hybrid` | `hybrid_pipeline` | `<stage> [--force-stage]` | `output/hybrid/`、`dataset/` | 任一阶段失败时非 0 |

`--reuse-existing-graph` 按稳定的来源 ID 重新绑定已保存的隐式依赖边,不调用模型。
`--candidate-id` 只校验包含指定候选的冲突。

### 数据集文件

除上述两层外,`build` 及之后的阶段还会在 `dataset/` 下写出审计文件:

| 文件 | 内容 |
|---|---|
| `quality_report.json` | `validate` 的结果:`ok`、失败的检查及对应条目、警告 |
| `source_ledger.json`、`page_coverage_ledger.json` | 哪些已登记对象被抽取到,分别在哪些页 |
| `source_grounding_report.json` | 每条记录在文本层上是如何定位的 |
| `formula_conflicts.json` | 各次转写之间的分歧及校验结果 |
| `transcription_review_queue.json`、`visual_audit_queue.json` | 需要人工查看的记录 |
| `source_span_collisions.json`、`canonical_text_overlaps.json` | 声称对应同一段文字的记录 |
| `superseded_entries.json`、`context_only_entries.json`、`id_aliases.json` | 被移除的重复项、仅作上下文的行、ID 重命名 |
| `dependency_graph.json` | 显式和隐式依赖边 |
| `symbol_glossary.json` | 记号表,以 `glossary_context` 挂到记录上 |
| `run_summary.json`、`benchmark_metrics.json` | 计数和运行元数据 |

## 配置

`run.json` 保存全部流水线设置,`init` 会生成一份可用的初始版本。所有相对路径以 `run.json` 所在目录为基准。
`vlm.base_url` 中的环境变量(如 `"${VLM_BASE_URL}"`)在加载配置时展开;变量未设置时直接报错。

主要配置段:

| 配置段 | 控制内容 |
|---|---|
| `vlm` | `base_url`、`model`、`api_key_env`、`timeout_seconds`、`request_attempts`、`segment_workers`、`vision_workers`、`double_pass` |
| `direct` | inventory、extraction、coverage audit、source verification、retry 的窗口大小、重叠页数、缩放、并发数、页范围 |
| `relationships` | `enabled`、`implicit_dependencies`、`workers`、`request_attempts`、批大小、候选上限、`fail_on_request_errors` |
| `prose_claims` | `enabled`、`span_grounding`、`anchors`、`subpart_policy`(`split`,或 `auto` 交给模型处理子问)、`granularity`(`atomize`、`record`、`atomize_merge_dependent`)、`claim_tiers`、`window_pages`、`overlap_pages`、`workers`、`output_suffix` |
| `mineru` | `backend`、`api_base`、`api_key_env`(默认 `MINERU_API_KEY`)、`model_version`、`cache_dir` |
| `pipeline` | 仅 MinerU 路线:`segment_chunk_chars`、`segment_overlap_pages`、`crop_padding`、`crop_zoom` |
| `hybrid` | 仅 hybrid 路线,见 [`docs/reference/hybrid-route.md`](docs/reference/hybrid-route.md) |

[`configs/run.example.json`](configs/run.example.json) 列出了每个键及其默认值。

`book_profile.json` 保存所有与具体书相关的设置,主要键有:
- `kinds` 和 `kind_aliases`;
- `label_regex`;
- `question_section_headings` 和 `support_section_headings`;
- `visual_prompt_overlay`;
- `exclude_pdf_page_ranges` 和 `glossary_page_ranges`;
- `drop_object_ids` 和 `merge_object_groups`;
- `source_verified_*` 系列覆盖项。

完整 schema:[`docs/reference/profile-schema.md`](docs/reference/profile-schema.md)。

**推荐的 `prose_claims` 设置**(来自 [`docs/experiments/ABLATION_RESULTS.md`](docs/experiments/ABLATION_RESULTS.md) 中的消融实验):

```json
"prose_claims": {
  "enabled": true,
  "span_grounding": true,
  "anchors": false,
  "subpart_policy": "split",
  "granularity": "atomize",
  "claim_tiers": false
}
```

`configs/run.example.json` 已采用这组默认值。

## 保证与不保证

流水线保证:

- `environments.json` 中的 `statement` 和 `supporting_text` 来自 VLM 对渲染页面的读取。文本层、MinerU
  或第二来源导出的文本只用于定位、排序和去重,绝不替换转写结果。
- 两次转写在公式上不一致时,`verify` 重读高倍裁剪图,并从已有候选中选出一个,不合成新的文本。
- 证明、提示和答案只在书中确实印出时才挂上(`supporting_text_origin = "book_source"`)。缺失的部分留空,
  不由模型生成。
- `relationships` 是事务式的:有请求错误时运行失败,原有的 `dependency_graph.json` 保持不变。
- `validate` 和 `preflight` 在检查失败时以退出码 2 退出,可用于脚本和 CI 把关。
- API key 只从环境变量读取。

流水线不保证:

- 完整召回。inventory、coverage audit 和 retry 能减少遗漏,但不能排除遗漏。覆盖率是相对检测轮次计算的,
  不是相对人工通读全书。
- 转写无误。流水线没有把握的记录会列入 `transcription_review_queue.json` 和 `visual_audit_queue.json`;
  不在这两个队列中的记录仍可能有错。
- `structured_statements.json` 忠实于原文。`normalized_statement` 由模型写出,其 verdict 是另一个模型的
  判断。规范化可能加入书中没有印出的假设。只需要原文时,请用 `source_statement` 或 `environments.json`。
- 无文本层扫描件的文本层定位。这类记录没有 `source_spans`,也不适用基于 span 的去重。

## 评测

评测工具在 `bench/`(见 [`bench/README.md`](bench/README.md)),共七个 case。每个 case 目录只提交
`run.json` 和 `book_profile.json`;PDF、gold 文件和输出含有受版权保护的文本,只保存在本地。

| 工具 | 衡量内容 |
|---|---|
| `bench/evaluate.py` | 对照 gold 单元:一对一召回率和精确率、文本层上的词级 span IoU(边界质量)、kind 准确率 |
| `bench/ablation_runner.py` | 运行 `VARIANTS` 中的 `prose_claims` 变体(各自使用独立缓存和输出后缀)并打分 |
| `bench/goldfree_eval.py report` | 无需 gold:卫生检查(控制字符、CJK 乱码、泄漏的引用/URL、幻觉前缀、空 statement、在来源页上找不到的证明)和印出 label 覆盖率 |
| `bench/goldfree_eval.py judge` | 对照页面图像,逐页盲评两个系统的输出,顺序随机 |

目前的 gold 单元由模型构建,而非人工标注。相同设置重复运行,召回率和 F1 最多相差约 0.1,因此小于约
±0.05 的差异需要多次重复才能判断。结果与分析见 [`docs/experiments/`](docs/experiments/)。

## 仓库结构

```
src/parse_ocr/
  cli.py                      parse-ocr 入口(命令 -> 模块映射)
  common.py                   配置加载、VLM 客户端、JSON 读写
  init_project.py             init
  preflight.py                preflight
  book_inspector.py           inspect
  direct_pipeline.py          direct 路线(主路线)
  source_grounding.py         build 使用的文本层定位
  text_spans.py               词级 span 定位与 IoU 工具
  verify_source_conflicts.py  verify
  extract_relationships.py    relationships
  prose_claims.py             prose-claims(结构化层)
  validate_dataset.py         validate
  mineru_pipeline.py          MinerU 路线
  hybrid_pipeline.py          hybrid 路线
configs/          run.example.json、book_profile.example.json
docs/reference/   schema 与各阶段设计说明
docs/experiments/ 消融实验计划、结果与报告
bench/            评测工具与各 case 配置(PDF、gold、输出不提交)
skill/            描述操作流程的 agent skill 封装
legacy/           早期基于 MinerU 的 HDP QA/VQA 脚本,仅供参考,不在测试范围内
tests/            单元测试(不联网,不调用 VLM)
```

## 开发

```bash
pip install -e '.[dev]'
pytest -q
```

测试不需要网络,也不需要 VLM。CI(`.github/workflows/tests.yml`)在 Python 3.9、3.10、3.12、3.13 上运行。

约定:
- 新增命令需要一个带 `main()`(解析 `sys.argv`)的模块,并在 `cli.COMMANDS` 中登记。
  `tests/test_cli_config.py` 会检查每个已登记模块都能导入且有 `main`。
- 调用模型的阶段在 `<output>/` 下写 checkpoint,除非传入 `--force-stage`,否则复用。新增阶段也应同样可续跑。
- 不要提交 PDF、gold 文件、输出、接口地址或密钥。

## 已知局限

- 含单纯形表的扫描书(Dantzig case)会丢失表格。
- 规范化可能加入书中没有印出的假设,校验器不一定能发现。
- 正文命题挖掘在习题密集的页面上读得不够全:习题之间的注记、记号和引述的事实常被漏掉(hdp_ex 上漏了 11 个单元)。
- 所有 gold 集都由模型构建。下一步评测的重点是对 `hdp_ex` 和 `perelman` 的 gold 做一轮人工校对。
