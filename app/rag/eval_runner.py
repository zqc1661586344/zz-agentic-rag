"""RAG 评测执行核心 — 生成 answer/contexts 并用 Ragas 打分。

供 scripts/eval_ragas.py（CLI）与 app/api/eval.py（子进程）复用。

流程：
  1. 读取评估集（GoldenDataset 表 enabled=1，或 JSON 文件）
  2. 对每条 question 调用 app.rag.chain.query_rag 得到 answer + chunks(contexts)
  3. 拼 Ragas 数据集（question/answer/contexts），用 Faithfulness 指标打分
  4. 返回 {metric: value}，由调用方写入 EvalRun

指标说明：默认只跑 Faithfulness（回答基于检索上下文的比例，0~1）。
它是无需 ground_truth、且对"检索/生成是否落地"最直接的信号；
Answer Relevancy / Context Recall 需要 ground_truth，作为可选扩展保留。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)


def load_testset(db=None, json_path: str | None = None, domain: str | None = None) -> list[dict]:
    """加载评估集：优先 JSON 文件；否则从 GoldenDataset 表读 enabled=1 数据。"""
    if json_path and Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "question": it["question"],
                "ground_truth": it.get("ground_truth", ""),
                "name": it.get("name"),
                "domain": it.get("domain"),
            }
            for it in data
            if it.get("question")
        ]

    if db is not None:
        from app.services.eval_service import list_golden

        rows = list_golden(db, domain=domain)
        return [
            {
                "question": r["question"],
                "ground_truth": "",
                "name": r.get("name"),
                "domain": r.get("domain"),
            }
            for r in rows
        ]

    return []


def _extract_chunks(result: dict) -> list[str]:
    """从 query_rag 返回里取上下文文本列表（chunks）。"""
    chunks = result.get("chunks") or []
    texts = []
    for c in chunks:
        content = c.get("content") if isinstance(c, dict) else getattr(c, "page_content", None)
        if content:
            texts.append(str(content))
    return texts


def run_eval(
    *,
    json_path: str | None = None,
    db=None,
    domain: str | None = None,
    top_k: int = 5,
    max_samples: int | None = None,
) -> dict:
    """执行一轮评估，返回 {metric: value}。缺数据/失败时抛异常供 CLI/API 捕获。"""
    testset = load_testset(db=db, json_path=json_path, domain=domain)
    if not testset:
        raise ValueError("评估集为空：请先导入 golden 数据或提供 --dataset 文件")

    if max_samples:
        testset = testset[:max_samples]

    from app.rag.chain import query_rag

    samples: list[dict] = []
    for it in testset:
        q = it["question"]
        logger.info("eval sample: %s", q[:40])
        result = query_rag(q, top_k=top_k)
        contexts = _extract_chunks(result)
        samples.append(
            {
                "question": q,
                "answer": str(result.get("answer", "")),
                "contexts": contexts,
                "ground_truth": it.get("ground_truth"),
            }
        )

    metrics = _score(samples)
    return metrics


def _ensure_ragas_importable() -> None:
    """ragas 0.2.x 顶层 import langchain_community.chat_models.vertexai，而该模块在
    langchain-community 0.4.x 已被移除（sunset）。评测只用 Faithfulness 且不会用
    VertexAI，这里在导入 ragas 前 stub 掉缺失模块，避免整个评测因不相关依赖崩溃。
    """
    import sys
    import types

    path = "langchain_community.chat_models.vertexai"
    if path in sys.modules:
        return
    mod = types.ModuleType(path)
    mod.ChatVertexAI = None
    mod.VertexAI = None
    mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[path] = mod


def _score(samples: list[dict]) -> dict:
    """用 Ragas 计算 Faithfulness。直接返回原始 float 值字典。"""
    if not samples:
        raise ValueError("无有效样本可打分")
    if all(not s["contexts"] for s in samples):
        # 完全无上下文的 free_chat 场景，Faithfulness 定义为 0
        logger.warning("所有样本均无检索上下文（free_chat），Faithfulness=0")
        return {"faithfulness": 0.0}

    _ensure_ragas_importable()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    # 用项目已配置的 LLM（同一 provider/base_url/key）作 judge，避免 ragas 默认
    # 从 OPENAI_API_KEY 读取而缺 key。test provider（假模型）不可评测真实文本。
    from app.config import settings
    from app.rag.llms import get_llm

    if settings.llm_provider == "test":
        raise ValueError("LLM_PROVIDER=test 无法执行真实 Ragas 评估，请配置真实 LLM")
    judge_llm = get_llm()
    if judge_llm is None:
        raise ValueError("LLM 未配置，无法执行 Ragas 评估打分")

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
            }
            for s in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        raise_exceptions=True,
        show_progress=False,
    )

    # ragas 0.2.x EvaluationResult：按指标名取 per-sample 分数列表
    try:
        scores = [float(x) for x in result["faithfulness"]]
    except (KeyError, TypeError):
        # 兜底：有的构建路径返回 dict / DataFrame
        cols = result.to_pandas() if hasattr(result, "to_pandas") else result
        try:
            scores = [float(x) for x in cols["faithfulness"].tolist()]
        except (KeyError, AttributeError):
            if isinstance(result, dict) and "faithfulness" in result:
                return {"faithfulness": round(float(result["faithfulness"]), 4)}
            raise

    if not scores:
        raise ValueError("Ragas 未返回 faithfulness 分数")
    avg = sum(scores) / len(scores)
    logger.info("eval faithfulness avg=%.4f (n=%d)", avg, len(scores))
    return {"faithfulness": round(avg, 4)}
