"""Agentic RAG 图编排与对外入口。

- run_agentic_retrieve：仅"检索决策阶段"（分析→检索→判分→纠偏循环），
  返回 {docs, free_chat}。供流式 query_rag_stream 在生成前并行出结果，保住流式协议。
- run_agentic：完整链路（检索决策 + 生成），供非流式 query_rag。
"""

import threading

from app.logging_config import get_logger

logger = get_logger(__name__)

# 编译后的图缓存（一次构建，跨查询复用）
_retrieval_graph = None
_full_graph = None
_graph_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 检索决策图专用终止节点
# ---------------------------------------------------------------------------


def _found(node_state: dict) -> dict:
    """判分相关：保留当前 docs，走生成。"""
    return {}


def _not_found(node_state: dict) -> dict:
    """反复检索未命中：转 free_chat，清空 docs/sources。"""
    return {"free_chat": True, "docs": [], "sources": []}


def _build_retrieval_graph():
    """编译『检索决策』图：analyze → retrieve → grade ──(relevant)→ found / (重试)→ re_retrieve→retrieve / (耗尽)→ not_found。"""
    from langgraph.graph import END, START, StateGraph

    from app.rag.agent.nodes import analyze_query, grade_relevance, re_retrieve, retrieve
    from app.rag.agent.state import AgentRAGState

    builder = StateGraph(AgentRAGState)
    builder.add_node("analyze", analyze_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_relevance)
    builder.add_node("re_retrieve", re_retrieve)
    builder.add_node("found", _found)
    builder.add_node("not_found", _not_found)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _route_retrieval,
        {"found": "found", "re_retrieve": "re_retrieve", "not_found": "not_found"},
    )
    builder.add_edge("re_retrieve", "retrieve")
    builder.add_edge("found", END)
    builder.add_edge("not_found", END)

    return builder.compile()


def _build_full_graph():
    """编译『完整』图：判分相关→generate，未命中→free_chat。"""
    from langgraph.graph import END, START, StateGraph

    from app.rag.agent.nodes import (
        analyze_query,
        free_chat,
        generate,
        grade_relevance,
        re_retrieve,
        retrieve,
    )
    from app.rag.agent.state import AgentRAGState

    builder = StateGraph(AgentRAGState)
    builder.add_node("analyze", analyze_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_relevance)
    builder.add_node("re_retrieve", re_retrieve)
    builder.add_node("generate", generate)
    builder.add_node("free_chat", free_chat)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _route_full,
        {"generate": "generate", "re_retrieve": "re_retrieve", "free_chat": "free_chat"},
    )
    builder.add_edge("re_retrieve", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("free_chat", END)

    return builder.compile()


def _route_retrieval(state: dict) -> str:
    """检索决策图的路由：相关→found；轮次耗尽→not_found；否则→纠偏。"""
    grade = state.get("grade") or {}
    if grade.get("is_relevant"):
        return "found"
    if (state.get("iteration") or 1) >= _max_rounds():
        return "not_found"
    return "re_retrieve"


def _route_full(state: dict) -> str:
    """完整图的路由：相关→generate；轮次耗尽→free_chat；否则→纠偏。"""
    grade = state.get("grade") or {}
    if grade.get("is_relevant"):
        return "generate"
    if (state.get("iteration") or 1) >= _max_rounds():
        return "free_chat"
    return "re_retrieve"


def _max_rounds() -> int:
    from app.config import settings

    return settings.rag_agentic_max_retrieval_rounds


def _get_retrieval_graph():
    global _retrieval_graph
    if _retrieval_graph is None:
        with _graph_lock:
            if _retrieval_graph is None:
                _retrieval_graph = _build_retrieval_graph()
    return _retrieval_graph


def _get_full_graph():
    global _full_graph
    if _full_graph is None:
        with _graph_lock:
            if _full_graph is None:
                _full_graph = _build_full_graph()
    return _full_graph


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def run_agentic_retrieve(
    query: str,
    top_k: int = 10,
    history: list[dict] | None = None,
    summary: str | None = None,
    user_id: str | None = None,
) -> dict:
    """仅运行检索决策阶段，返回 {docs, free_chat}。供流式/非流式共用。"""
    graph = _get_retrieval_graph()
    result = graph.invoke(
        {
            "query": query,
            "history": history,
            "summary": summary,
            "user_id": user_id,
            "plan": {"mode": "retrieve", "top_k": top_k},
            "iteration": 0,
        }
    )
    return {
        "docs": result.get("docs") or [],
        "free_chat": bool(result.get("free_chat")),
    }


def run_agentic(
    query: str,
    top_k: int = 10,
    history: list[dict] | None = None,
    summary: str | None = None,
    user_id: str | None = None,
    history_text: str = "",
) -> dict:
    """完整链路（检索决策 + 生成），返回 {answer, sources, chunks, free_chat}。"""
    graph = _get_full_graph()
    result = graph.invoke(
        {
            "query": query,
            "history": history,
            "summary": summary,
            "history_text": history_text,
            "user_id": user_id,
            "plan": {"mode": "retrieve", "top_k": top_k},
            "iteration": 0,
        }
    )
    free_chat = bool(result.get("free_chat"))
    docs = result.get("docs") or []
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources") or [],
        "chunks": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
        "free_chat": free_chat,
    }
