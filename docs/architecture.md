# 企业级 Agentic RAG 系统 — 技术方案文档

> 版本：2.0.0 · 2026-10-23
> 本文描述**现行**架构，与代码仓库现状保持一致。
> 快速开始与验收路径见 [agentic_rag_plan.md](./agentic_rag_plan.md)。

---

## 目录

- [企业级 Agentic RAG 系统 — 技术方案文档](#企业级-agentic-rag-系统--技术方案文档)
  - [目录](#目录)
  - [1. 架构总览](#1-架构总览)
  - [2. 技术栈选择](#2-技术栈选择)
  - [3. 系统模块划分](#3-系统模块划分)
  - [4. 数据模型](#4-数据模型)
  - [5. 鉴权与多租户数据边界](#5-鉴权与多租户数据边界)
  - [6. REST API 设计](#6-rest-api-设计)
    - [6.1 认证 `/api/auth`](#61-认证-apiauth)
    - [6.2 文档 `/api/documents`](#62-文档-apidocuments)
    - [6.3 对话 / 检索 `/api/conversations`](#63-对话--检索-apiconversations)
    - [6.4 评测 `/api/eval`](#64-评测-apieval)
    - [6.5 其他](#65-其他)
  - [7. 前端 Streamlit](#7-前端-streamlit)
  - [8. 后台任务（Celery / BackgroundTasks 双轨）](#8-后台任务celery--backgroundtasks-双轨)
  - [9. Agentic RAG 检索链路](#9-agentic-rag-检索链路)
    - [9.1 检索内核（确定性）](#91-检索内核确定性)
    - [9.2 Agent 决策层（LangGraph）](#92-agent-决策层langgraph)
    - [9.3 与 chain.py 的接线](#93-与-chainpy-的接线)
  - [10. 离线评测（Golden Dataset + Ragas）](#10-离线评测golden-dataset--ragas)
  - [11. 核心配置项（.env）](#11-核心配置项env)
    - [11.1 Agentic RAG](#111-agentic-rag)
    - [11.2 检索内核](#112-检索内核)
    - [11.3 数据 / 后台](#113-数据--后台)
    - [11.4 评测相关](#114-评测相关)
  - [12. 启动与部署](#12-启动与部署)
    - [12.1 本地开发](#121-本地开发)
    - [12.2 生产部署要点](#122-生产部署要点)
  - [13. 生产注意事项](#13-生产注意事项)
  - [附录 A：依赖清单](#附录-a依赖清单)
    - [核心依赖](#核心依赖)
    - [可选依赖（uv extras）](#可选依赖uv-extras)
  - [附录 B：验证方案](#附录-b验证方案)
  - [附录 C：源码文件速查](#附录-c源码文件速查)
- [附录 A：依赖清单](#附录-a依赖清单)
- [附录 B：验证方案](#附录-b验证方案)
- [附录 C：源码文件速查](#附录-c源码文件速查)

---

## 1. 架构总览

系统采用 **「确定性混合检索内核 + LangGraph Agent 决策层」** 相结合的 Agentic RAG 架构：

- **检索内核（确定性）**：稠密向量（PGVector）+ 稀疏全文（PG tsvector）双路召回，RRF 融合，可选 bge-reranker 重排；叠加 Multi-Query 多路召回。此层算法**稳定可复现**，是与评测基线对齐的"地基"。
- **决策层（Agent，LangGraph）**：在检索结果之上做"路由 + 自纠偏"——分析改写查询 → 判分检索相关度 → 不足则闭环重检索（最多 N 轮）→ 耗尽则降级自由聊天。**不引入多跳规划 / 全工具自主选择**，保持最小可用与可观测。
- **自由聊天回退**：文档检索未命中时，明确声明"当前已有文档中找不到答案"，转 LLM 自身知识回答，避免幻觉与误导。

组件连接示意：

```
Browser/Streamlit (8002)
      │  SSE / REST
      ▼
FastAPI (8001)
      │  POST /api/conversations/{id}/query  (Flow: /query, /query/stream)
      ▼
┌─────────────────────────── Agentic RAG 层 ───────────────────────────┐
│  query_rag / query_rag_stream                                         │
│        │  rag_agentic_enabled ?                                       │
│        ├─ NO → 旧确定性漏斗（并行回退，仅当开关关闭时走）            │
│        └─ YES → LangGraph 检索决策图 + 现有生成链                      │
│                 analyze → retrieve → grade ──相关──→ 生成             │
│                   ↑          │          └─不足→ re_retrieve (≤N轮)    │
│                   └──────────┴──耗尽→ free_chat                       │
└───────────────────────────────────────────────────────────────────────┘
      │  docs / free_chat
      ▼
检索内核: dense(PGVector) + sparse(pg_tsvector) + RRF + Multi-Query [+ rerank]
      │
      ▼
PostgreSQL(业务表 + pgvector + tsvector GIN)   Redis(Celery broker + 缓存)
```

**关键设计原则**

- **接口签名守恒**：对外 `query_rag` 返回 `{answer, sources, chunks, free_chat}`、流式事件协议 `[free_chat]? → token* → sources([DONE])` 在 Agentic 化前后**完全不变**，前端零改动。
- **一键回退**：`RAG_AGENTIC_ENABLED=false` 时 100% 走旧确定性漏斗，Agent 层不参与。
- **分层解耦**：`agent/` 包内节点通过函数体内惰性 import 复用现有检索内核与生成链，避免与 `chain.py` 形成循环依赖。
- **最小 LLM 开销**：`RAG_AGENTIC_USE_MULTIQUERY=false` 默认单路检索，判分结构化输出失败时降级为"信任检索结果"。

---

## 2. 技术栈选择

| 层次 | 选型 | 说明 |
|------|------|------|
| Web / API | FastAPI 0.115+ · Uvicorn · Gunicorn | 高性能、原生 async、OpenAPI 文档 |
| 前端 | Streamlit 1.40+ | 快速构建对话 + 文档管理界面 |
| 数据库 | PostgreSQL 14+ · SQLAlchemy 2.0 · Alembic | 业务数据 + 向量（pgvector）+ 全文（tsvector）同库承载 |
| 向量检索 | PGVector（`langchain-postgres`） | collection `documents`，`bge-m3` 1024 维，cosine 度量 |
| 稀疏检索 | PG `tsvector` + GIN 索引（默认） | 替代内存 BM25，天然持久化、支持 SQL 过滤 |
| ORM | SQLAlchemy 2.0（sync） | 简化开发；满仓同步 + 少量异步 |
| RAG 框架 | LangChain 1.x · `langchain-classic` · `langchain-postgres` | 检索内核、文档加载、向量库集成 |
| Agent 编排 | LangGraph 1.x | 检索决策图（路由 + 自纠偏）与离线评测 Agent |
| 异步任务 | Celery 5.4 + Redis（broker），回退 BackgroundTasks | 文档处理、离线评测 |
| LLM / Embedding | `langchain-openai`（默认）/ `langchain-ollama` / `test` | 支持 OpenAI 兼容端点；`test` 用于开发与 CI |
| 重排 | bge-reranker-v2-m3（siliconflow 在线 / local） | 可选，用于精排 |
| 评测 | `ragas` + `datasets` + LLM Judge | Golden Dataset 离线评测、回归对比 |
| 限流 | `slowapi` | 登录接口 30 次/分钟等 |
| 认证 | JWT（python-jose）+ passlib/bcrypt | 三角色：admin / editor / viewer |

> 说明：原 Chroma + 内存 BM25 已**替换**为 PGVector + PG tsvector；通用工作流平台（`workflows`）已**移除**，业务流程统一由 LangGraph Agent 承担。

---

## 3. 系统模块划分

```
app/
├── main.py               # FastAPI 入口：lifespan 初始化(建表/种子/FST索引/恢复卡死文档/provider探针)
├── config.py             # 全局配置（pydantic-settings + .env）
├── database.py           # 引擎/Session、建表与默认角色种子
├── celery_app.py         # Celery 实例（broker = CELERY_BROKER_URL）
├── logging_config.py     # 日志配置
├── dependencies.py       # 鉴权依赖：get_current_user / require_roles
├── models/               # SQLAlchemy ORM：user/document/conversation/eval
├── schemas/              # Pydantic 请求/响应模型
├── api/                  # 路由：auth/users/documents/conversations/eval/health
├── services/             # 业务服务：auth/document/rag_service/eval_service
├── cache/                 # Redis 客户端（复用 broker URL，DB2 隔离）
├── rag/
│   ├── chain.py          # RAG 查询链入口、Agentic 开关分支、自由聊天回退、对话摘要
│   ├── retrievers.py     # 稠密+稀疏双路检索、RRF 融合、Multi-Query、rerank 封装
│   ├── vector_store.py   # PGVector 封装（collection "documents"）
│   ├── sparse_search.py  # pg tsvector 检索 + GIN 索引幂等建索
│   ├── rerankers.py      # bge-reranker 封装（siliconflow/local）
│   ├── embeddings.py     # Embedding 初始化（OpenAI/Ollama/Test）
│   ├── llms.py           # LLM 初始化（OpenAI/Ollama/Test）
│   ├── splitter.py       # 文本分块配置
│   ├── pipeline.py       # 文档处理管线（解析→分块→索引→FTS 同步）
│   ├── tasks.py          # Celery 文档处理任务
│   ├── eval/             # 离线评测 agent/指标/Dataset 注入
│   └── agent/            # Agentic 检索决策层
│       ├── state.py      # LangGraph State（total=False 全可选）
│       ├── prompts.py    # 判分(structured)/纠偏改写提示词
│       ├── tools.py      # 薄封装现有检索内核（惰性 import 防循环）
│       ├── nodes.py      # analyze/retrieve/grade(结构化布尔)/re_retrieve/generate/free_chat
│       └── graph.py      # 编译"检索决策图"/"完整图"，对外 run_agentic_retrieve/run_agentic
└── streamlit_app/        # 前端：app.py + api_client.py + auth.py
```

---

## 4. 数据模型

| 表 | 说明 | 关键字段 |
|----|------|---------|
| `users` | 用户 | username/email/hashed_password、`is_superuser`、`is_active` |
| `roles` | 角色 | admin / editor / viewer（启动时种子） |
| `user_roles` | 用户-角色多对多 | (user_id, role_id) |
| `documents` | 文档元数据 | uploaded_by、status(pending/processing/indexed/failed)、chunk_count |
| `document_chunks` | 分块 + PGVector 向量 + tsvector | `page_content`、`embedding(1024d)`、`search_text tsvector`（GIN 索引进）、metadata |
| `conversations` | 对话会话 | created_by、title、summary、context_preview |
| `messages` | 对话消息 | role、content、sources(JSON 文本)、created_at |
| `eval_runs` | 评测运行记录 | scene、trigger、params、metric_values、status、regression/treshold |
| `golden_datasets` | 离线评测数据集 | 描述、条目数 |

**FTS 索引（仅 PG，幂等）**：`document_chunks.search_text` 上的 GIN 索引起动时由 `sparse_search.ensure_fts_index()` 保证存在。

---

## 5. 鉴权与多租户数据边界

- **认证**：`POST /api/auth/register|login|refresh`，返回 JWT；`/me`、`/change-password`。登录接口受 slowapi 限流。
- **角色**：注册用户默认 `viewer`；`admin`（is_superuser）由 `scripts/create_admin.py` 创建；`roles` 表启动时种子化三角色。
- **数据隔离（多租户）**：
  - 文档 `uploaded_by` 标识所有者；非 superuser 仅能操作自己的文档，superuser 可见全部。
  - 检索内核按 `user_id` 过滤：普通用户只能检索"私有文档 + 全部共享文档"；superuser 使用 `"__all__"` 全量索引/全量检索。
  - 检索入口通过登录用户 ID 传递，服务端强制隔离，前端不可伪造。
- **鉴权依赖**：`get_current_user`（Bearer Token）、`require_roles(admin, editor)` 等限制敏感/管理操作。

---

## 6. REST API 设计

### 6.1 认证 `/api/auth`
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /register | 注册（默认 viewer）+ 返回访问/刷新令牌 |
| POST | /login | 登录（限流） |
| POST | /refresh | 刷新令牌 |
| GET | /me | 当前用户信息（含角色） |
| POST | /change-password | 修改密码 |

### 6.2 文档 `/api/documents`
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | / | 上传文档（流式写盘，大文件安全） |
| GET | / | 列文档（按用户/共享过滤） |
| GET | /{id} | 详情（含 chunk 预览） |
| DELETE | /{id} | 删除（级联删 chunk + 失效索引/缓存） |
| POST | /{id}/reprocess | 重新索引 |

文档处理统一走 `services/rag_service.enqueue_process()`：按配置走 Celery `delay()` 或 `BackgroundTasks.add_task()`。

### 6.3 对话 / 检索 `/api/conversations`
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | / | 建会话 |
| GET | /{id} | 取会话（owner/superuser） |
| POST | /{id}/query | 同步问答：`{answer, sources, chunks, free_chat}` |
| POST | /{id}/query/stream | **SSE 流式**问答 |
| GET | /{id}/messages | 消息列表 |
| DELETE | /{id} | 删除会话 |

**SSE 事件协议（Agentic 前后守恒）**：
```
[free_chat 事件（仅未命中时，先发）]
token 事件 × N      # "token" : 增量文本
sources 事件         # "sources" : [{...}]，附带文档引用与 [Source N] 标记
DONE
```

### 6.4 评测 `/api/eval`
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /run | 触发评测（scene: rag/all），Celery 或子进程 |
| GET | /history | 历史与回归对比 |

### 6.5 其他
- `GET /api/health`：存活检查（readiness 对 DB/PGVector 实时 probe；LLM/Embedding 用启动时缓存探针）。

---

## 7. 前端 Streamlit

- 入口 `app/streamlit_app/app.py`，`api_client.py` 封装 REST/SSE 调用，`auth.py` 管理登录态。
- 两个前端页面：**对话（Agentic RAG 问答）** 与 **文档管理**。
- 通过后端 `/api/auth` 登录获取 JWT，后续请求携带 Bearer Token；SSE 增量渲染 `[Source N]` 引用。

---

## 8. 后台任务（Celery / BackgroundTasks 双轨）

- 统一入口 `services/rag_service.enqueue_process()`：`CELERY_BROKER_URL` 非空且 `USE_CELERY_TASK=True` 时走 Celery `delay()`，否则回退 `BackgroundTasks.add_task()`。
- Celery worker：`celery -A app.celery_app worker`，`autodiscover(["app.rag"])` 注册 `rag/tasks.py` 的文档处理任务（持久化、可重试）。
- 作用范围：文档 ingestion、离线评测（`eval.py` 内同样双轨判断）。
- `app/main.py` lifespan 启动时调用 `_recover_stuck_documents()`，恢复上次异常退出时卡在 `processing` 的文档。

---

## 9. Agentic RAG 检索链路

### 9.1 检索内核（确定性）

`retrievers.py` 提供，作为 Agent 工具与旧漏斗共用：

- **稠密检索**：`vector_store.py`（PGVector）cosine 相似度，`relevance = 1 - dist`。
- **稀疏检索**：`sparse_search.py`（pg tsvector + GIN），支持关键词匹配。
- **融合**：双路 top-k 用 `_rrf_fuse()` 做 RRF 融合（受 `RAG_HYBRID_ALPHA` 加权）。
- **Multi-Query**：当 `RAG_MULTI_QUERY_ENABLED=true`，N 个变体查询并行多路召回（`RAG_MULTI_QUERY_MAX_WORKERS` 控并发），结果去重。
- **可选重排**：`RAG_RERANK_ENABLED=true` 时用 bge-reranker（siliconflow 在线或本地）交叉编码器重排，取 `RAG_RERANK_TOP_N`。
- **阈值**：`hybrid` 模式通过 `min_score` + `min_spread` 判定"命中检索"还是"回退自由聊天"。

### 9.2 Agent 决策层（LangGraph）

`rag/agent/graph.py` 编译两张图（一次构建，跨查询缓存复用，线程安全）：

**检索决策图**（供流式入口并行出结果，保住流式协议）：
```
analyze(改写/分析) → retrieve(调检索内核) → grade(结构化判分)
                                                  │
                相关(relevant)───→ found(走生成)
                不相关且轮次未耗尽 → re_retrieve(纠偏改写) → retrieve(重搜)
                不相关且轮次耗尽 → not_found(free_chat)
```

**完整图**：检索决策 + `generate`（复用现有轻量生成链）或 `free_chat`。

对外入口（`graph.py`，threading.Lock 保护图缓存）：
- `run_agentic_retrieve(query, history, user_id) → {docs, free_chat}` —— 仅决策，供流式使用。
- `run_agentic(query, history, user_id) → {answer, sources, chunks, free_chat}` —— 完整链路，供非流式。

节点说明（`nodes.py`）：
| 节点 | 职责 |
|------|------|
| `analyze_query` | 复用 `_rewrite_query` 做查询分析/改写，注入历史上下文 |
| `retrieve` | 调 `tools.hybrid_search_tool` / `multi_query_search_tool` 拉 top_k 文档 |
| `grade_relevance` | **结构化布尔判分**：`is_relevant: bool + reason + score`（中文防注入提示词）；结构化输出失败降级为"信任检索" |
| `re_retrieve` | 按判分 reason 纠偏改写 query，自增 iteration（受 `RAG_AGENTIC_MAX_RETRIEVAL_ROUNDS` 上限防死循环） |
| `generate` / `free_chat` | 复用现有链生成带引用答案或自由聊天 |

### 9.3 与 chain.py 的接线

- `query_rag` / `query_rag_stream` 入口处：`if settings.rag_agentic_enabled:` 走 Agent 分支；否则走旧确定性漏斗（完全保留于原位）。
- 流式：先 `run_agentic_retrieve` 拿到 `{docs/free_chat}`，再复用现有链流式生成 token，事件顺序 `[free_chat]? → token* → sources([DONE])` 不变。

---

## 10. 离线评测（Golden Dataset + Ragas）

- **数据**：`scripts/gen_eval_dataset.py` 生成 Golden Dataset，存 `golden_datasets` 表。
- **执行**：`rag/` 内评测 Agent 用 Ragas 指标（如 answer correctness、faithfulness 等）对检索+生成打分；`eval_service.run_and_persist_eval()` 落库 `eval_runs` 并与基线对比，产出回归判定与退出码。
- **触发**：`POST /api/eval/run`（scene=`rag`/`all`，Celery 或后台子进程）或 CLI。
- **参数**：记录 `top_k / search_type / multi_query / llm / embedding / min_score`，保证可复现。

---

## 11. 核心配置项（.env）

### 11.1 Agentic RAG
| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_AGENTIC_ENABLED` | true | 启用 Agent 层；false → 完全回退旧确定性漏斗 |
| `RAG_AGENTIC_MAX_RETRIEVAL_ROUNDS` | 2 | 自纠偏最大重检索轮数（防死循环） |
| `RAG_AGENTIC_GRADE_ENABLED` | true | 是否做相关性判分 |
| `RAG_AGENTIC_USE_MULTIQUERY` | false | Agent 决策层内是否启用多查询（默认单路，省 token） |

### 11.2 检索内核
| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_SEARCH_TYPE` | hybrid | similarity / mmr / hybrid |
| `RAG_HYBRID_ALPHA` | 0.3 | 稠密 vs 稀疏 RRF 权重 |
| `RAG_HYBRID_MIN_SPREAD` | 0.015 | hybrid 下 top-1 与 top-2 最小分差 |
| `RAG_MIN_SCORE` | 0.4 | 相关性绝对阈值 |
| `RAG_SPARSE_BACKEND` | pg_tsvector | bm25_memory / pg_tsvector |
| `RAG_MULTI_QUERY_ENABLED` | true | 多查询多路召回开关 |
| `RAG_MULTI_QUERY_N` | 3 | 变体查询数 |
| `RAG_MULTI_QUERY_PRIMARY_WEIGHT` | 0.5 | 原始查询权重 |
| `RAG_MULTI_QUERY_TOP_K_SCALE` | 2.0 | 各路 top_k 放大 |
| `RAG_MULTI_QUERY_MAX_WORKERS` | 4 | 并行 worker 上限 |
| `RAG_RERANK_ENABLED` | false | 交叉编码器重排开关 |
| `RAG_RERANK_PROVIDER` | siliconflow | local / siliconflow |
| `RAG_RERANK_MODEL` | BAAI/bge-reranker-v2-m3 | 重排模型 |
| `RAG_RERANK_TOP_N` | 5 | 重排后保留 top-n |
| `VECTOR_COLLECTION_NAME` | documents | PGVector collection |
| `RAG_EMBEDDING_DIM` | 1024 | bge-m3 向量维度 |

### 11.3 数据 / 后台
| 变量 | 默认 | 说明 |
|------|------|------|
| `DATABASE_URL` | sqlite:///... | 熟测可 SQLite；**生产务必 PostgreSQL**（pgvector/tsvector 仅 PG） |
| `CELERY_BROKER_URL` | — | Redis，如 `redis://localhost:6379/0`；空 → 回退 BackgroundTasks |
| `CELERY_RESULT_BACKEND` | — | Celery 结果后端 |
| `USE_CELERY_TASK` | false | 强制使用 Celery |
| `RATE_LIMIT_ENABLED` | true | 登录等接口限流 |
| `UPLOAD_DIR` / `MAX_UPLOAD_SIZE_MB` | ./data/uploads / 50 | 上传配置 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | 分块参数 |
| `APP_DEBUG` / `LOG_LEVEL` | true / INFO | 日志 |

### 11.4 评测相关
| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_EVAL_REGRESSION_THRESHOLD` | — | 回归判定阈值 |
| `RAG_TOP_K` | 5 | 评测记录 top_k |

---

## 12. 启动与部署

### 12.1 本地开发
```bash
uv sync
bash start.sh          # 后端 8001 + Streamlit 8002
# admin 账号：
python scripts/create_admin.py
```
- 后端 `http://localhost:8001`，API 文档 `/docs`；前端 `http://localhost:8002`。
- 开发默认 `sqlite` + `test` LLM（或填 OPENAI_API_KEY）。
- 需要 Celery + Redis：先 `redis-server`，再 `celery -A app.celery_app worker --loglevel=info`，确保 `CELERY_BROKER_URL` 已设置。

### 12.2 生产部署要点
- 数据库：PostgreSQL（数据库 URL 为 `postgresql://...`）。
- 进程：`gunicorn -k uvicorn.workers.UvicornWorker app.main:app`（main.py 内 `uvicorn.run` reload 分支仅供开发）。
- Celery worker 常驻；Redis 作为 broker/缓存。
- `RAG_AGENTIC_ENABLED` 演验证后保持 true；出问题时快速 `false` 回退旧漏斗。

---

## 13. 生产注意事项

- **向量与全文依赖 PG**：pgvector、pg tsvector GIN 需要 PostgreSQL，生产不要用 SQLite。
- **FTS/向量索引**：启动幂等建索；文档增删后 chunk 与向量/tsvector 同步清理。
- **判分鲁棒性**：Agent 判分对结构化输出失败已做"信任检索"降级，不对 LLM 结果强依赖。
- **防死循环**：重检索轮数受 `RAG_AGENTIC_MAX_RETRIEVAL_ROUNDS` 上限约束。
- **日志**：通过 `APP_DEBUG` / `LOG_LEVEL` 控制；hybrid 判定与 Agent 路由输出到日志便于排查；建议生产集成结构化日志轮转。
- **限流**：登录等敏感接口启用 slowapi 限流；高频调试可 `RATE_LIMIT_ENABLED=false`。

---

## 附录 A：依赖清单

### 核心依赖
| 包 | 版本 | 用途 |
|----|------|------|
| fastapi / uvicorn / gunicorn | — | Web 框架 / ASGI / WSGI |
| sqlalchemy / alembic | ≥2.0 / ≥1.13 | ORM / 迁移 |
| pydantic / pydantic-settings | ≥2.0 | 校验 / 配置 |
| python-jose / passlib[bcrypt] | — | JWT / 密码哈希 |
| slowapi / python-multipart / python-dotenv / httpx | — | 限流 / 上传 / 配置 / HTTP |
| langchain / langchain-community | ≥1.3, <2 | RAG |
| langchain-openai | ≥1.4 | LLM/Embedding（OpenAI 兼容） |
| langchain-classic | ≥1.0.8 | EnsembleRetriever/重排 |
| langchain-postgres | ≥0.0.17 | PGVector 集成 |
| langgraph / langgraph-checkpoint-postgres | ≥0.2 / ≥3.1 | Agent 图编排 / 检查点 |
| rank-bm25 / jieba | — | 稀疏检索回退 + 中文分词 |
| sqlalchemy pg 驱动 / psycopg2-binary | — | PostgreSQL |
| celery / redis | ≥5.4 / ≥5.0 | 异步任务 / broker |
| pypdf / python-docx / docx2txt / beautifulsoup4 / openpyxl / python-pptx | — | 多格式文档解析 |
| datasets / ragas | ≥5.0 / ≥0.2.15 | 离线评测 |

### 可选依赖（uv extras）
| extra | 包 | 用途 |
|-------|------|------|
| postgres | asyncpg | 异步 PG 驱动 |
| ollama | langchain-ollama | Ollama 本地集成 |
| langgraph | langgraph, langgraph-checkpoint-postgres, psycopg[binary] | Agent 生产检查点 |
| rerank | transformers, torch | bge-reranker 本地重排（默认不安装，装易因本地编译失败） |
| dev | pytest, pytest-asyncio, ruff, mypy, ragas, datasets | 开发/测试 |

---

## 附录 B：验证方案

| 阶段 | 验证方法 | 预期结果 |
|------|---------|---------|
| 基础设施 | `bash start.sh` | 8002 显示 Streamlit UI；8001/docs 显示 Swagger |
| 认证 | `create_admin.py` 建 admin → 登录 → /me | 收到 JWT，返回 admin 角色 |
| 文档管理与索引 | 上传文档 → 查看状态 | 变为 indexed，chunk_count > 0；PG 中有向量 + tsvector |
| Agentic RAG 问答 | 输入与文档相关的问题 | 走 检索决策图→生成，返回带 `[Source N]` 引用的回答 |
| 自纠偏（多轮） | 输入较模糊、首轮判分不足的问题 | 日志可见 re_retrieve（迭代 1..N）后命中并正确回答 |
| 自由聊天回退 | 输入无关 query（如"你好"） | 收到"当前已有文档中找不到答案"前缀回答，无来源，SSE 先发 free_chat |
| 开关回退 | `RAG_AGENTIC_ENABLED=false` 重启 | 100% 走旧确定性漏斗，接口结构/事件协议不变 |
| SSE 协议 | 前端流式问答 | `[free_chat]? → token* → sources([DONE])` 顺序正确 |
| 检索模式切换 | 修改 `RAG_SEARCH_TYPE` / `RAG_RERANK_ENABLED` | 检索行为相应变化 |
| 数据隔离 | 普通用户 A 查不到另一用户私有文档；superuser 可见全部 | 检索结果按 user_id 正确过滤 |
| 后台任务 | 配置 Celery 后上传文档 | 文档由 worker 异步索引，失败可重试 |
| 离线评测 | `POST /api/eval/run` | eval_runs 有记录、与基线可对比 |
| 文档删除 | 删除已索引文档 | chunk 级联删除，检索中不再出现 |
| 端到端 | `python test_e2e.py` | 全部通过 |

---

## 附录 C：源码文件速查

| 文件 | 职责 |
|------|------|
| `app/rag/chain.py` | RAG 查询链入口、Agentic 开关分支、自由聊天回退、对话摘要 |
| `app/rag/retrievers.py` | 稠密+稀疏双路检索、RRF 融合、Multi-Query、可选 rerank |
| `app/rag/vector_store.py` | PGVector 封装（collection "documents"，cosine） |
| `app/rag/sparse_search.py` | pg tsvector 检索 + GIN 幂等建索 |
| `app/rag/rerankers.py` | bge-reranker 封装（siliconflow / local） |
| `app/rag/pipeline.py` | 文档处理管线（解析→分块→索引→FTS 同步） |
| `app/rag/embeddings.py` | Embedding 初始化（OpenAI/Ollama/Test） |
| `app/rag/llms.py` | LLM 初始化（OpenAI/Ollama/Test） |
| `app/rag/splitter.py` | 文本分块配置 |
| `app/rag/tasks.py` | Celery 文档处理任务 |
| `app/rag/agent/graph.py` | Agentic 检索决策图 / 完整图，run_agentic_retrieve / run_agentic |
| `app/rag/agent/nodes.py` | analyze / retrieve / grade(结构化布尔) / re_retrieve / generate / free_chat |
| `app/rag/agent/tools.py` | 薄封装检索内核（惰性 import） |
| `app/rag/agent/prompts.py` | 判分 / 纠偏改写提示词 |
| `app/rag/agent/state.py` | LangGraph State 定义 |
| `app/config.py` | 全局配置（pydantic-settings + .env） |
| `app/api/conversations.py` | 对话 CRUD、历史记忆、RAG 查询入口（同步 + SSE 流式） |
| `app/api/documents.py` | 文档上传/列表/详情/删除/重处理 |
| `app/api/eval.py` | 评测触发与历史 |
| `app/services/rag_service.py` | `enqueue_process` 统一 Celery/BackgroundTasks 入口 |
| `app/services/document_service.py` | 文档 CRUD + 索引失效同步 |