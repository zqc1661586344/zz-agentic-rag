"""检索能力薄封装 —— 复用现有检索内核，不重写算法。

供 agent 图节点调用；函数体内惰性 import 以避免与 chain.py 形成循环依赖。
"""

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def hybrid_search_tool(query: str, top_k: int = 5, user_id: str | None = None) -> list[Document]:
    """稠密 + 稀疏 RRF 混合检索（复用 retrievers.hybrid_search）。"""
    from app.rag.retrievers import hybrid_search

    return hybrid_search(query, top_k=top_k, user_id=user_id)


def multi_query_search_tool(
    query: str, top_k: int = 5, user_id: str | None = None
) -> list[Document]:
    """多视角多路召回（复用 retrievers.multi_query_search）。"""
    from app.rag.retrievers import multi_query_search

    return multi_query_search(query, top_k=top_k, user_id=user_id)


def retrieve_top_docs(
    query: str, top_k: int = 5, user_id: str | None = None, use_multiquery: bool = True
) -> list[Document]:
    """按开关选择单路 hybrid 或多路 multi-query。"""
    if use_multiquery:
        return multi_query_search_tool(query, top_k=top_k, user_id=user_id)
    return hybrid_search_tool(query, top_k=top_k, user_id=user_id)
