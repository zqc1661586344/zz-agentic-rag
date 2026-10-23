# 企业 RAG 智能问答系统（Agentic RAG）

> **版本**: 0.1.0
> **最后更新**: 2026-10-23
> **项目状态**: Agentic RAG

一套面向企业的 **Agentic RAG** 智能问答系统：

- 💬 **RAG 智能问答** — 上传内部文档，LangGraph 编排「分析路由 → Hybrid 检索 → 结构化判分 → 纠偏重检索」自我修正链路，SSE 流式回答
- ⚙️ **混合检索内核** — 稠密 PGVector + 稀疏 PG tsvector/BM25 + RRF 融合 + 可选重排（Multi-Query 多路召回）

**前端** Streamlit（对话 + 文档管理）| **后端** FastAPI（8001）| **数据** SQLite / PostgreSQL | **编排** LangGraph | **包管理** uv

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.11 | |
| uv | 最新 | 包管理 |
| PostgreSQL | ≥ 15（可选） | 生产推荐，开发可用 SQLite |

### 一键启动

```bash
# 1. 克隆 & 进入
git clone <repo> && cd zz-agentic-rag

# 2. 安装依赖
uv sync

# 3. 复制并填写配置
cp .env.example .env
# 至少需设置：LLM_API_KEY / EMBEDDING_API_KEY / JWT_SECRET_KEY

# 4. 启动（后端 8001 + Streamlit 前端 8002）
bash start.sh
# → 后端 http://localhost:8001
# → 前端 http://localhost:8002
# → API 文档 http://localhost:8001/docs
```

### 全新库初始化 admin

首次启动自动建表（`Base.metadata.create_all` + Alembic）。生产环境不会自动种 admin，需要手动启用一次：

```bash
SEED_DEMO_USER=true bash start.sh
# → 用 admin / admin123 登录后立即改密
curl -X POST http://localhost:8001/api/auth/change-password \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"old_password":"admin123","new_password":"<强密码>"}'
# → 重启（不带开关）
bash start.sh
```

---

## 🏗️ 系统架构

```mermaid
flowchart TD
    subgraph 用户层
        SL["Streamlit 前端 (8002)"]
        API_CLI["API 客户端 (curl / SDK)"]
    end

    subgraph "FastAPI 网关 (8001)"
        ROUTER["路由层 (app/api/router.py)"]
        JWT["JWT 鉴权 + 限流中间件"]
    end

    subgraph "RAG 子系统"
        AUTH["Auth / Users"]
        DOCS["Documents 管理 + 解析管线"]
        CONV["Conversations + RAG 查询"]
        AGENT["Agentic RAG (LangGraph)<br/>analyze → retrieve → grade<br/>→ (自纠偏) re_retrieve → generate / free_chat"]
        RETRIEVER["Hybrid Retriever<br/>Multi-Query + (PGVector 稠密 + tsvector/BM25 稀疏 + RRF + 重排)"]
    end

    subgraph "异步调度层"
        CELERY["Celery Task Queue<br/>(Redis broker)<br/>文档索引"]
    end

    subgraph "数据层"
        DB["SQLite / PostgreSQL<br/>Alembic 迁移"]
        VEC_DOC["PGVector: documents 集合<br/>(bge-m3 1024d)"]
        FILES["文件存储<br/>(原始文档落盘)"]
    end

    SL --> ROUTER
    API_CLI --> ROUTER
    ROUTER --> JWT
    JWT --> AUTH & DOCS & CONV

    DOCS --> CELERY
    CELERY --> AGENT
    CONV --> AGENT
    AGENT --> RETRIEVER
    RETRIEVER --> VEC_DOC
    RETRIEVER --> DB

    DOCS --> FILES
    DOCS --> VEC_DOC & DB
```

---

## 📋 使用流程

### RAG 问答流程

```mermaid
flowchart LR
    A["注册 / 登录"] --> B["上传文档<br/>支持 PDF/DOCX/MD 等"]
    B --> C["等待索引完成<br/>解析 → 分块 → 向量化"]
    C --> D["新建对话"]
    D --> E["提问"]
    E --> F["Agentic RAG<br/>LangGraph 自纠偏检索"]
    F --> G["LLM 生成回答<br/>带来源引用"]
```

### Streamlit 页面结构

| 页面 | 功能 |
|------|------|
| 💬 对话 | RAG 问答 + 流式回答 + 多轮历史 |
| 📁 文档管理 | 上传 / 删除 / 索引状态查看 |

---

## 📄 支持的文件格式

系统共用同一套文档解析管线，支持以下格式：

| 格式 | 扩展名 | MIME 类型 |
|------|--------|-----------|
| PDF | `.pdf` | `application/pdf` |
| 纯文本 | `.txt` | `text/plain` |
| Markdown | `.md` | `text/markdown` |
| Word 文档 | `.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| CSV | `.csv` | `text/csv` |
| HTML | `.html` | `text/html` |
| Excel | `.xlsx` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| PowerPoint | `.pptx` | `application/vnd.openxmlformats-officedocument.presentationml.presentation` |
| TOML | `.toml` | `application/toml` |

---

## 🔎 Agentic RAG 检索流程

系统默认采用 **Agentic RAG**：以 **LangGraph** 编排「分析路由 → 检索 → 结构化判分 → 纠偏重检索」的自纠偏链路，检索内核仍是成熟的 **Hybrid RAG**（稠密 PGVector + 稀疏 PG tsvector/BM25 + RRF 融合 + 可选重排）。

```mermaid
flowchart TD
    Q["用户提问"] --> H["组装对话历史（最近 20 轮 + 更早摘要）"]
    H --> A["analyze<br/>多轮改写 → rewrite_query"]
    A --> RV["retrieve<br/>Hybrid 检索（可切 Multi-Query 多路）"]
    RV --> G{"grade<br/>LLM 结构化判分（布尔）"}
    G -->|相关| GN["generate<br/>format_context + RAG_PROMPT"]
    GN --> L["LLM 生成"]
    L --> OUT["SSE 流式输出 + [Source N] 引用"]
    G -->|不相关 且&nbsp;已到最大轮次| FC["free_chat<br/>纯 LLM 回答，不附带来源"]
    G -->|不相关 且&nbsp;未到最大轮次| RR["re_retrieve<br/>据判分反馈改写查询"]
    RR --> RV
    FC --> OUT
```

### 核心设计要点

- **Agent 编排（LangGraph）**：图节点 `analyze → retrieve → grade → (re_retrieve → retrieve) → generate / free_chat`；`grade` 用结构化布尔相关性做 CRAG 判分，不相关且未超最大检索轮数时 `re_retrieve` 依据判分反馈改写查询重检索，**天然防死循环**（受 `RAG_AGENTIC_MAX_RETRIEVAL_ROUNDS` 限制）。
- **开关与回退**：`RAG_AGENTIC_ENABLED`（默认 `true`）启用 Agentic 流程；设为 `false` 时 100% 回退旧确定性漏斗（`chain.py` 原生保留），便于 A/B 对比。
- **混合检索内核**：稠密向量（PGVector + bge-m3）+ 稀疏关键词（PG tsvector + jieba 中文分词，旧版/回退用内存 BM25）双路 `hybrid_search`，RRF 融合排序，可选 bge-reranker 交叉编码器重排。
- **Hybrid 三种检索模式**：`hybrid`（默认）/ `similarity` / `mmr`，通过 `RAG_SEARCH_TYPE` 切换。
- **Multi-Query 多路召回**（hybrid 模式默认开启，由 `RAG_MULTI_QUERY_ENABLED` 配置）：把问题用 LLM 扩为 `RAG_MULTI_QUERY_N` 个视角，各视角独立检索后按 `RAG_MULTI_QUERY_PRIMARY_WEIGHT` 做二级 RRF 融合；free-chat 判定以原问题（首路）为准。Agentic 流程下默认用单路（`RAG_AGENTIC_USE_MULTIQUERY=false`）控延迟成本，可在配置中开启。
- **无命中回退**：判分不相关且检索轮次耗尽 → 回退 `自由聊天`（`free_chat=true`，提示语由前端按标记渲染，不附带来源）。
- **文档生命周期**：上传时对每个 chunk 用 jieba 分词写入 `search_text`，删除时随行删除——增量维护，无需全量重建索引；旧库启动时由 `ensure_fts_index()` 自动补列 + 建 GIN 索引。

> 📄 **详细流程**：完整检索结构、Multi-Query、RRF 融合公式、重排器配置见 [`docs/RAG.md`](docs/RAG.md)；Agentic RAG 改造方案见 [`docs/agentic_rag_plan.md`](docs/agentic_rag_plan.md)；端到端架构见 [`docs/architecture.md`](docs/architecture.md)。

---

## 🔌 API 概览

### 认证

```bash
# 注册
curl -X POST http://localhost:8001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"Secure@123","email":"alice@example.com","full_name":"Alice"}'

# 登录
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"Secure@123"}'
# → 返回 {access_token, refresh_token}
```

### 文档

```bash
# 上传文档
curl -X POST http://localhost:8001/api/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@report.pdf"

# 查看文档状态
curl http://localhost:8001/api/documents/<doc-id> \
  -H "Authorization: Bearer <token>"
# → status: "indexed" 表示索引完成

# 列出所有文档
curl http://localhost:8001/api/documents \
  -H "Authorization: Bearer <token>"
```

### RAG 问答

```bash
# 创建对话
curl -X POST http://localhost:8001/api/conversations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"title":"产品咨询"}'

# 提问（RAG 查询，默认走 Agentic 流程）
curl -X POST http://localhost:8001/api/conversations/<conv-id>/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query":"产品的核心功能有哪些？"}'
# → 返回 {answer, sources: [{filename, page}], free_chat}
```

完整 API 端点清单见启动后访问 `/docs`，或 [`docs/architecture.md`](docs/architecture.md)。

---

## 🐳 生产部署

### 1. 更换密钥

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

将生成的密钥填入 `.env` 的 `JWT_SECRET_KEY`。**生产禁止使用默认值**。

### 2. 启动服务

**方式一：start.sh（推荐）**

```bash
bash start.sh
# 后端 8001 + Streamlit 8002 同时启动
```

**方式二：分别后台启动**

```bash
nohup .venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 --port 8001 > /tmp/rag.log 2>&1 &

nohup .venv/bin/streamlit run app/streamlit_app/app.py \
  --server.port 8002 --server.headless true > /tmp/streamlit.log 2>&1 &

# 验证
curl http://localhost:8001/api/health
# → {"status":"ok","version":"0.4.0"}
```

### 3. 迁移到 PostgreSQL

```bash
uv sync --extra postgres
# .env
DATABASE_URL=postgresql://user:password@host:5432/rag_db
alembic upgrade head
```

### 4. Nginx 反向代理

```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;

    # 后端 API + SSE
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Streamlit 前端（WebSocket 必需）
    location /streamlit/ {
        proxy_pass http://127.0.0.1:8002/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 5. systemd 服务管理

**后端** `/etc/systemd/system/rag-backend.service`：

```ini
[Unit]
Description=RAG Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/zz-demand-system
ExecStart=/opt/zz-demand-system/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
Restart=always
RestartSec=5
Environment=APP_DEBUG=false

[Install]
WantedBy=multi-user.target
```

**前端** `/etc/systemd/system/rag-frontend.service`：

```ini
[Unit]
Description=Streamlit Frontend
After=network.target
Requires=rag-backend.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/zz-demand-system
ExecStart=/opt/zz-demand-system/.venv/bin/streamlit run app/streamlit_app/app.py --server.port 8002 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 🧪 测试

### 端到端

```bash
LLM_PROVIDER=test EMBEDDING_PROVIDER=test .venv/bin/uvicorn app.main:app --port 8001 &
python test_e2e.py
```

覆盖场景：
- 用户注册 → 登录 → 获取个人信息
- 上传文档 → 索引完成
- 创建对话 → RAG 查询 → 获取消息
- 健康检查

### Agentic 检索决策图单测

```bash
# 编译两张 LangGraph 图 + 路由逻辑校验（不触发 LLM/检索）
LLM_PROVIDER=test uv run python -c "from app.rag.agent.graph import _get_retrieval_graph, _get_full_graph; _get_retrieval_graph(); _get_full_graph(); print('graph OK')"
```

---

## 📁 项目结构

```
zz-agentic-rag/
├── app/
│   ├── api/                        # FastAPI 路由层
│   ├── middleware/                  # CORS / 限流 / Tracing
│   ├── models/                     # SQLAlchemy ORM（auth/doc/conversation 等）
│   ├── schemas/                    # Pydantic 请求/响应模型
│   ├── services/                   # RAG 业务逻辑（rag_service 门面层）
│   ├── rag/                        # RAG 核心
│   │   ├── agent/                  #   Agentic RAG（LangGraph）
│   │   │   ├── graph.py            #    图编排：run_agentic / run_agentic_retrieve
│   │   │   ├── nodes.py            #    analyze/retrieve/grade/re_retrieve/generate/free_chat
│   │   │   ├── tools.py            #    检索能力薄封装
│   │   │   ├── prompts.py          #    判分/纠偏提示词
│   │   │   └── state.py            #    图状态定义
│   │   ├── chain.py                #    RAG 编排（Agentic 开关 + 旧漏斗回退）
│   │   ├── retrievers.py           #    Hybrid 检索（Multi-Query + RRF）
│   │   ├── vector_store.py         #    PGVector 稠密检索
│   │   ├── sparse_search.py        #    PG tsvector 稀疏检索
│   │   ├── rerankers.py            #    可选 bge-reranker
│   │   ├── llms.py                 #    LLM 工厂（OpenAI/Ollama/test）
│   │   ├── pipeline.py             #    摄入管线
│   │   └── splitters.py / loaders.py / tokenizer.py
│   ├── cache/                      # Redis 缓存
│   ├── celery_app.py               # Celery 应用（文档索引任务）
│   ├── streamlit_app/              # Streamlit 前端
│   │   ├── app.py                  #   入口 + 页面路由
│   │   ├── api_client.py           #   后端 HTTP 客户端（含 token 管理）
│   │   ├── auth.py                 #   登录态管理
│   │   └── views/
│   │       ├── chat.py              #     💬 对话
│   │       ├── documents.py        #     📁 文档管理
│   │       └── login.py            #     登录页
├── alembic/                        # 数据库迁移
├── docs/                           # 设计文档
│   ├── architecture.md             #   端到端技术方案
│   ├── RAG.md                      #   Hybrid RAG 详解
│   └── agentic_rag_plan.md         #   Agentic RAG 改造方案
├── data/                           # 运行时落盘
├── start.sh                        # 一键启动脚本
├── pyproject.toml                  # uv + 依赖管理（含 langgraph）
└── alembic.ini
```

---

## 🔧 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI 0.115+ | 异步 + OpenAPI 自动生成 |
| **前端** | Streamlit 1.40+ | 纯 Python UI（对话 + 文档管理） |
| **ORM** | SQLAlchemy 2.0+ | |
| **迁移** | Alembic | |
| **认证** | JWT（python-jose）+ bcrypt | RBAC 角色：admin / editor / viewer |
| **向量库** | PGVector（bge-m3 1024d） | collection：`documents` |
| **混合检索** | PG tsvector（默认）/ rank_bm25（回退）+ jieba | RRF 融合 |
| **可选重排** | bge-reranker-v2-m3 | 需 transformers + torch（默认关） |
| **LLM / Embedding** | OpenAI 兼容协议 / Ollama / test mock | 配置化切换 |
| **RAG 框架** | LangChain 1.3+ | |
| **Agent 编排** | LangGraph | 自纠偏检索图 + 状态管理 |
| **文档解析** | PyPDF / python-docx / docx2txt / markdown | |
| **包管理** | uv | |
| **中间件** | slowapi（限流）+ Tracing（request_id）+ CORS | |

---

## 📈 未来规划

### RAG 核心
- [x] 流式输出（SSE）
- [x] OpenAPI 兼容第三方供应商（DeepSeek / 通义千问 / Ollama）
- [x] Hybrid RAG（PGVector + tsvector + RRF）
- [x] Agentic RAG（LangGraph 路由 + 自纠偏）
- [ ] 批处理文档导入（文件夹上传）
- [ ] 文档预览与在线编辑
- [ ] 检索测试工具（给定问题查看原始检索结果）
- [ ] Agentic RAG 增强：多跳规划 / web 兜底 / 工具自主选择
- [ ] LangSmith 观测 agent 决策路径

---

## 📝 License

MIT