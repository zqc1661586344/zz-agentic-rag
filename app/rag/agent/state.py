"""Agentic RAG 图状态定义（LangGraph State）。"""

from typing import Any, TypedDict


class AgentRAGState(TypedDict, total=False):
    """LangGraph 图中流转的状态。

    - 检索决策阶段由模型驱动（分析/判分/纠偏重检索）；
    - 生成阶段复用现有轻量链，保证流式协议与 [Source N] 引用不变。
    """

    # 输入与上下文
    query: str  # 原始问题（首轮用户问题）
    history: list[dict] | None  # 多轮对话历史（raw，供改写）
    summary: str | None  # 更早对话摘要（供改写/生成）
    history_text: str  # format_history 结果（供生成提示词）
    user_id: str | None

    # 分析/改写产物
    rewrite_query: str  # 当前用于检索的查询（改造后/纠偏后）
    plan: dict[str, Any]  # 检索策略（mode 等）

    # 检索与判分
    iteration: int  # 已检索轮数（含首轮，防死循环）
    docs: list[Any]  # 当前轮检索结果（Document 列表）
    grade: dict[str, Any] | None  # {is_relevant, reason, score}

    # 输出
    free_chat: bool  # 是否走自由对话
    answer: str
    sources: list[dict[str, Any]]
