#!/usr/bin/env python3
"""RAG 评测 CLI — 被子进程（app/api/eval.py）调用，亦可手动运行。

用法:
    python scripts/eval_ragas.py run rag --top-k 5
    python scripts/eval_ragas.py run rag --top-k 5 --threshold 0.05
    python scripts/eval_ragas.py run rag --dataset ./datasets/compliance_rag_eval.json

流程:
  1. 加载评估集（GoldenDataset 表 / --dataset 文件）
  2. run_eval：对每条 question 调 query_rag 生成 answer+contexts，Ragas 打 Faithfulness 分
  3. run_and_persist_eval：写入 EvalRun，输出回归判定
退出码: 0 = 成功/无回归；1 = 有回归；2 = 执行失败
"""

from __future__ import annotations

import argparse
import os
import sys

# 使 `python scripts/eval_ragas.py` 能导入项目根下的 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_scene(args: argparse.Namespace) -> int:
    from app.database import SessionLocal
    from app.rag.eval_runner import run_eval
    from app.services.eval_service import run_and_persist_eval

    db = SessionLocal()
    try:
        try:
            metrics = run_eval(
                json_path=args.dataset,
                db=db,
                domain=getattr(args, "domain", None),
                top_k=args.top_k,
                max_samples=getattr(args, "max_samples", None),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[eval] 评估执行失败: {e}", file=sys.stderr)
            run_and_persist_eval(
                scene=args.scene,
                metric_values={},
                dataset_name=args.dataset,
                trigger="cli",
                notes=f"eval failed: {e}",
                failed=True,
            )
            return 2

        result = run_and_persist_eval(
            scene=args.scene,
            metric_values=metrics,
            dataset_name=args.dataset,
            trigger="cli",
            threshold=args.threshold,
        )
        print(f"\n[eval] 完成 run_id={result['run_id']} status={result['status']}")
        print(f"[eval] 指标: {metrics}")
        if result["regressions"]:
            print(f"[eval] ⚠️  回归: {result['regressions']}")
            return 1
        print("[eval] ✅ 无回归")
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测（Ragas Faithfulness）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="运行一轮评估")
    run_p.add_argument("scene", choices=["rag", "all"], help="评估场景")
    run_p.add_argument("--top-k", type=int, default=5, help="检索 top_k")
    run_p.add_argument("--threshold", type=float, default=None, help="回归阈值 (0~1)")
    run_p.add_argument(
        "--dataset", type=str, default=None, help="JSON 评估集文件（缺省用 GoldenDataset 表）"
    )
    run_p.add_argument("--domain", type=str, default=None, help="按 domain 过滤 golden 数据")
    run_p.add_argument("--max-samples", type=int, default=None, help="最多评估 N 条（调试用）")
    run_p.set_defaults(func=run_scene)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
