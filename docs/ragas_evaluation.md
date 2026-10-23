# RAG 离线评估指南（Ragas + Golden Dataset）

本文档描述如何使用项目内置的 **离线自动化评估**，对 **Agentic RAG 问答链路**进行量化评估、持久化与回归对比。

当前评估形态为 **「Golden Dataset + Ragas 指标 + API/CLI 触发 + 回归判定」**，取消了对合规审查（compliance）的专项评估场景——系统已聚焦单一 RAG/Agentic 引擎，评估场景只有 `rag` 与 `all`（后者等于 rag）。

---

## 一、评估架构

```
GoldenDataset 表（golden_datasets："题目 + ground_truth"）
      │  gen_eval_dataset.py 可从法规 seed_data 生成
      ▼
Ragas 评估（answer reliability 等指标，LLM-as-judge）
      │  app/api/eval.py 触发（POST /api/eval/run）
      ▼
EvalRun 表（eval_runs：一次评估快照：params + metric_values + status）
      │  eval_service 落库
      ▼
GET /api/eval/history → 回归对比（相对最近基线，判定是否回落）
```

- **持久化**：不等于临时一次性脚本——每次评估写入 `eval_runs`，保留最近 `RAG_EVAL_KEEP_RECENT`（默认 30）轮，可回溯趋势。
- **回归判定**：`eval_service` 取最近 `RAG_EVAL_BASELINE_WINDOW`（默认 1）次的指标均值作基线，当某指标下降绝对值超过 `RAG_EVAL_REGRESSION_THRESHOLD`（默认 0.05）即判定回归，供 CI 或人工决策。
- **可复现**：每次运行快照当前关键参数 `top_k / search_type / multi_query / llm / embedding / min_score`。

---

## 二、数据模型

| 表 | 说明 | 关键字段 |
|----|------|---------|
| `golden_datasets` | 固定评估集条目 | name、question、ground_truth、domain、enabled |
| `eval_runs` | 一次评估快照 | trigger（manual/cli）、scene（rag）、dataset_name、params（JSON）、metric_values（JSON）、status（success/failed）、notes |

场景枚举：`scene ∈ {"rag", "all"}`（`all` 与 `rag` 等同，保留用于扩展）。

---

## 三、生成评估集

### 内置数据
`scripts/eval_dataset_labor.json` 提供一组 **劳动合同/劳动法方向** 的 question + ground_truth。

**为什么是劳动合同？** 项目默认业务域为劳动/合同法，预期上传到知识库的第一批文档也是劳动合同法、民法典等法规。若知识库没有对应文档，评估会走 `free_chat`（检索为空），Faithfulness ≈ 0——恰好暴露"知识库缺文档"。

### 自动生成
`scripts/gen_eval_dataset.py` 从法规 seed 数据（`**/*.json`）逐条调用 LLM，生成 3-5 条自然语言 `question + ground_truth`，输出 JSON 直接可作为评估集：

```bash
python scripts/gen_eval_dataset.py
python scripts/gen_eval_dataset.py --per-article 5 --output ./datasets/compliance_rag_eval.json
python scripts/gen_eval_dataset.py --max-articles 20
```

### 自建评估集（推荐）
JSON 数组格式，每项 `{question, ground_truth}`：

```json
[
  { "question": "试用期最长不能超过多长时间？",
    "ground_truth": "劳动合同期限三个月以上不满一年的，试用期不得超过三个月..." },
  { "question": "竞业限制的期限最长是多久？",
    "ground_truth": "竞业限制的期限不得超过二年。" }
]
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `question` | ✅ | 用户会提的真实问题 |
| `ground_truth` | 推荐 | 理想参考答案，用于 Answer Relevancy / Context Recall |

**设计建议**：固定一组 20-50 条 query 用于回归；覆盖精确匹配 / 模糊 / 超范围 / 多跳推理等边界 case；每次改动检索参数后跑同一组才能比较。

---

## 四、触发评估

### 方式 1：REST API（后台异步，推荐）
```bash
# 触发一次评估（admin）
curl -X POST http://localhost:8001/api/eval/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"scene": "rag", "top_k": 5}'

# 查看结果与历史
curl http://localhost:8001/api/eval/history?scene=rag \
  -H "Authorization: Bearer $TOKEN"
```

- 评估以**后台子进程**方式执行（复用 `scripts/eval_ragas.py`），与 Celery 无关，无需依赖 broker。
- 子进程失败（非零退出/异常/超时 `RAG_EVAL_SUBPROCESS_TIMEOUT`）会自动补写一条 `status=failed` 的 `EvalRun`，history 中可见明确失败原因。
- 默认使用 GoldenDataset 表（`enabled=1`）作评估集；如需指定 JSON 文件，见下方「方式 2」。

### 方式 2：CLI（离线纯命令行）
独立执行入口 `scripts/eval_ragas.py`，与 API 共用同一套评估核心（`app/rag/eval_runner.py` + `eval_service`），适合手动跑一轮、调试或 CI 回归。

流程：加载评估集 → 对每条 question 调 `query_rag` 生成 answer+contexts → Ragas Faithfulness 打分 → 写入 `EvalRun` 并输出回归判定。

```bash
# 用 JSON 文件作评估集（推荐：可控、可切片）
uv run python scripts/eval_ragas.py run rag \
  --dataset scripts/eval_dataset_labor.json --top-k 5

# 用 GoldenDataset 表（enabled=1）作评估集
uv run python scripts/eval_ragas.py run rag --top-k 5

# 回归阈值 + 调试用少量样本
uv run python scripts/eval_ragas.py run rag \
  --dataset scripts/eval_dataset_labor.json --top-k 3 \
  --threshold 0.05 --max-samples 5
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `scene` | — | `rag` / `all` |
| `--top-k` | 5 | 检索 top_k |
| `--threshold` | 取 `RAG_EVAL_REGRESSION_THRESHOLD` | 回归阈值（0~1），超阈值触发退出码 1 |
| `--dataset` | 用 GoldenDataset 表 | JSON 评估集文件路径 |
| `--domain` | 全部 | 按 domain 过滤 GoldenDataset |
| `--max-samples` | 全部 | 最多评估 N 条（调试用） |

退出码：`0` 成功且无回归；`1` 有回归；`2` 执行失败（此时会补写一条 `status=failed` 的 `EvalRun`）。

---

## 五、核心指标

> 当前评估默认只启用 **Faithfulness**（与检索落地直接相关、轻量、不依赖 ground_truth）；下表其余指标为 ragas 能力，如需可扩展。

| 指标 | 含义 | 分数低说明 |
|------|------|-----------|
| **Faithfulness** ✅默认 | 回答中有多少完全基于检索上下文 | LLM 在瞎编（幻觉） |
| **Answer Relevancy** | 回答真正回答问题的部分占比 | 答非所问 / 跑题 |
| **Context Recall** | 应检索到的内容实际拿到多少 | 检索漏召回 |
| **Context Precision** | 检索到的内容中多少是真相关 | 检索噪声多 |

Ragas 是 LLM-as-judge：每项指标额外要 1-3 次 LLM 调用，样本越多耗时越长（20 条 query 约 8-12 分钟）。

**分数速查（RAG）**

| 区间 | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|------|-------------|-----------------|----------------|-------------------|
| 0.9-1.0 | 几乎无幻觉 | 直击要点 | 该检索的都拿到了 | 几乎无噪声 |
| 0.7-0.9 | 偶有轻度幻觉 | 大部分相关 | 漏召回 <30% | 噪声 <30% |
| 0.5-0.7 | 频繁幻觉 | 部分答非所问 | 漏召回严重 | 噪声较多 |
| <0.5 | 基本瞎编 | 几乎跑题 | 检索基本没命中 | 基本全是噪声 |

---

## 六、典型使用场景

1. **参数调优**：改 `chunk_size` / `RAG_SEARCH_TYPE` / `RAG_RERANK_ENABLED` 后重跑同一评估集，看 Context Recall / Precision 是否提升（对比 `eval_runs` 历史）。
2. **RAG 参数回归**：改了 `retrievers.py` 的 RRF 权重 / 阈值后跑一次，确认未退化（基线对比由 `eval_service` 自动判定）。
3. **Agentic 层回归**：改 `rag/agent/` 判分/纠偏后跑一次，重点看 Faithfulness 与 Context Precision 是否因自纠偏/路由改变而波动。

---

## 七、相关配置（.env）

| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_EVAL_TOP_K` | 5 | 评估检索 top_k |
| `RAG_EVAL_SUBPROCESS_TIMEOUT` | 1800 | 评估子进程超时（秒），LLM 逐样本生成+打分不宜设过小 |
| `RAG_EVAL_KEEP_RECENT` | 30 | EvalRun 保留最近轮数，过期自动清理 |
| `RAG_EVAL_BASELINE_WINDOW` | 1 | 回归基线取最近 N 次均值；默认 1=最近一次 |
| `RAG_EVAL_REGRESSION_THRESHOLD` | 0.05 | 指标相对基线下降超过该值（绝对值 0~1）判定回归 |

---

## 八、常见问题

**Q: Faithfulness 全部为 0？**
**A**: 两条可能：① 知识库为空，query 走 `free_chat`，contexts 为空 → Faithfulness 必然 0。先上传文档。② query 完全超纲，换评估集匹配已有文档业务域。

**Q: 评估很慢？**
**A**: Ragas 每指标每样本要 1-3 次 LLM 调用。样本量越大越慢，评估是低频操作；可减少样本或用更少的指标。

**Q: /api/eval/run 返回 queued 但 history 看不到成功记录？**
**A**: 检查后台子进程是否报错（history 会写 `status=failed` + notes 说明原因），确认知识库非空、LLM/Embedding 可用。评测以子进程运行，不依赖 Celery。

---

## 九、相关文件

| 文件 | 说明 |
|------|------|
| `app/api/eval.py` | 评估 API：`POST /api/eval/run`、`GET /api/eval/history` |
| `app/rag/eval_runner.py` | 评测核心：加载评估集 → query_rag → Ragas Faithfulness 打分 |
| `scripts/eval_ragas.py` | CLI 入口（被 `api/eval.py` 子进程复用） |
| `app/services/eval_service.py` | 评估落库 + 基准回归判定 |
| `app/models/eval.py` | GoldenDataset / EvalRun ORM |
| `scripts/gen_eval_dataset.py` | 从法规 seed 自动生成评估集 |
| `scripts/eval_dataset_labor.json` | 内置劳动合同评估集 |
| `pyproject.toml`（dev extra） | ragas + datasets 版本锁定 |