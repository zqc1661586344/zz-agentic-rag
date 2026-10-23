"""Agentic RAG 图节点 —— 检索决策阶段（路由 / 检索 / 判分 / 纠偏 / 生成 / 自由对话）。

节点的生成部分复用 chain.py 的现有轻量链（RAG_PROMPT / FREE_CHAT_PROMPT），
保证生成逻辑、[Source N] 引用与流式协议与旧漏斗完全一致；仅前置"检索决策"被 agent 化。
"""

from pydantic import BaseModel, Field

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 结构化判分 Schema
# ---------------------------------------------------------------------------
class GradeResult(BaseModel):
    """判分结构化输出。"""

    is_relevant: bool = Field(description="检索到的片段是否足以支撑回答")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性分数 0~1")
    reason: str = Field(default="", description="一句话说明")


# ---------------------------------------------------------------------------
# 节点实现
# ---------------------------------------------------------------------------


def analyze_query(state: dict) -> dict:
    """分析节点：多轮改写 + 设定默认检索计划。

    路由决策不在此处用额外的 LLM 预先分流到自由对话（保留旧语义："free_chat 只能在
    检索判分后仍未命中时触发"），避免为常规问题增加一次不必要的 LLM 调用。
    """
    from app.rag.chain import _rewrite_query

    raw_query = state.get("query", "")
    rewrite = _rewrite_query(raw_query, state.get("history"), state.get("summary", None))
    logger.info("agentic analyze: raw=%r → rewrite=%r", raw_query, rewrite)

    return {
        "rewrite_query": rewrite,
        "iteration": state.get("iteration", 0),
        "plan": {"mode": "retrieve"},
        "docs": [],
        "grade": None,
    }


def retrieve(state: dict) -> dict:
    """检索节点：按计划调用现有检索内核，迭代轮次 +1。"""
    from app.rag.agent.tools import retrieve_top_docs

    query = state.get("rewrite_query") or state.get("query", "")
    top_k = int(state.get("plan", {}).get("top_k", 10) or 10)
    use_multiquery = settings.rag_agentic_use_multiquery

    docs = retrieve_top_docs(
        query, top_k=top_k, user_id=state.get("user_id"), use_multiquery=use_multiquery
    )
    return {
        "docs": docs,
        "iteration": state.get("iteration", 0) + 1,
    }


def grade_relevance(state: dict) -> dict:
    """判分节点（CRAG）：结构化判分检索结果是否相关。

    - 未检索到文档 → is_relevant=False（触发纠偏/自由对话）；
    - 判分关闭 → 信任检索（有文档即视为相关，退化为旧漏斗行为）；
    - 打开判分 → LLM 结构化布尔判分；解析失败降级为"信任检索"。
    """
    query = state.get("rewrite_query") or state.get("query", "")
    docs = state.get("docs") or []
    context_lines = [d.page_content for d in docs if hasattr(d, "page_content")]
    context = "\n\n".join(context_lines)

    if not context:
        return {
            "grade": {
                "is_relevant": False,
                "score": 0.0,
                "reason": "未检索到任何相关内容，需要纠偏重检索或转为自由对话",
            }
        }
    if not settings.rag_agentic_grade_enabled:
        return {"grade": {"is_relevant": True, "score": 1.0, "reason": "判分关闭，信任检索结果"}}

    grade = _llm_grade(query, context)
    logger.info(
        "agentic grade: query=%r relevant=%s score=%s reason=%s",
        query[:50],
        grade.is_relevant,
        grade.score,
        grade.reason[:60],
    )
    return {"grade": grade.model_dump()}


def _llm_grade(query: str, context: str) -> GradeResult:
    """LLM 结构化判分；不可用时返回默认（信任检索）。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from app.rag.agent.prompts import GRADE_HUMAN, GRADE_SYSTEM
    from app.rag.llms import get_llm

    llm = get_llm()

    # 尝试结构化输出（openai/兼容协议、ollama 的 with_structured_output）
    try:
        struct_llm = llm.with_structured_output(GradeResult)
        return struct_llm.invoke({"query": query, "context": context})
    except Exception:
        logger.debug("structured grade failed, fall back to text parse", exc_info=True)

    # 降级：文本生成 → 正则解析；解析失败默认信任检索（is_relevant=True）
    try:
        from langchain_core.runnables import RunnablePassthrough

        prompt = ChatPromptTemplate.from_messages(
            [("system", GRADE_SYSTEM), ("human", GRADE_HUMAN)]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"query": query, "context": context}) or ""
        return _parse_grade_text(raw)
    except Exception:
        logger.warning("grade fell back to default (trust retrieval)", exc_info=True)
        return GradeResult(is_relevant=True, score=1.0, reason="判分异常，信任检索结果")


def _parse_grade_text(raw: str) -> GradeResult:
    """从文本中解析 is_relevant / score / reason。"""
    import re

    text = raw.strip()
    relevant = "不相关" not in text and 'is_relevant": false' not in text.replace(" ", "")
    score_m = re.search(r"score[\"':\s]*([\d.]+)", text, re.IGNORECASE)
    score = float(score_m.group(1)) if score_m else (1.0 if relevant else 0.0)
    reason_m = re.search(r"reason[\"':\s]+(.+)", text, re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else ""
    return GradeResult(is_relevant=bool(relevant), score=score, reason=reason)


def re_retrieve(state: dict) -> dict:
    """纠偏节点：依据判分反馈改写查询，供下一轮检索。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from app.rag.agent.prompts import REWRITE_HUMAN, REWRITE_SYSTEM
    from app.rag.llms import get_llm

    query = state.get("rewrite_query") or state.get("query", "")
    reason = (state.get("grade") or {}).get("reason", "检索结果与问题不相关")
    try:
        prompt = ChatPromptTemplate.from_messages(
            [("system", REWRITE_SYSTEM), ("human", REWRITE_HUMAN)]
        )
        chain = prompt | get_llm() | StrOutputParser()
        rewrite = (chain.invoke({"query": query, "reason": reason}) or "").strip()
        rewrite = rewrite or query
    except Exception:
        logger.warning("agentic re_retrieve rewrite failed, keep original", exc_info=True)
        rewrite = query
    logger.info("agentic re_retrieve: %r → %r", query, rewrite)
    return {"rewrite_query": rewrite}


def generate(state: dict) -> dict:
    """生成节点：用现有 RAG 链生成带 [Source N] 的答案并清洗引用。"""
    from app.rag.chain import build_rag_chain, format_context, format_sources, sanitize_citations

    docs = state.get("docs") or []
    query = state.get("query", "")
    history_text = state.get("history_text", "")

    context = format_context(docs)
    sources = format_sources(docs)

    chain = build_rag_chain()
    answer = chain.invoke(
        {"context": context, "question": query, "history": history_text},
        config={"tags": ["rag", "agentic"]},
    )
    answer = sanitize_citations(answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "free_chat": False,
    }


def free_chat(state: dict) -> dict:
    """自由对话节点：未命中知识库时用纯 LLM 回答，不附带来源。"""
    from app.rag.chain import _build_free_chat_chain

    query = state.get("query", "")
    history_text = state.get("history_text", "")
    chain = _build_free_chat_chain()
    answer = chain.invoke(
        {"question": query, "history": history_text},
        config={"tags": ["agentic_free_chat"]},
    )
    return {"answer": answer, "sources": [], "free_chat": True}


# ---------------------------------------------------------------------------
# 条件路由
# ---------------------------------------------------------------------------


def route_after_grade(state: dict) -> str:
    """判分后的条件边：相关→生成；轮次耗尽→自由对话；否则→纠偏重检索。"""
    grade = state.get("grade") or {}
    is_relevant = bool(grade.get("is_relevant"))
    iteration = state.get("iteration", 1)
    max_rounds = settings.rag_agentic_max_retrieval_rounds

    if is_relevant:
        return "generate"
    if iteration >= max_rounds:
        logger.info("agentic: not relevant after %d rounds → free_chat", iteration)
        return "free_chat"
    return "re_retrieve"
