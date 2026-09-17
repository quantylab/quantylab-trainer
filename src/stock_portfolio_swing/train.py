"""Train and locked-evaluate the stock-portfolio-swing product."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

from .dataset import load_stock_portfolio_frame
from .portfolio import backtest_portfolio, fit_forward_model, make_forward_target, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock portfolio swing 지도학습")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default="stock_20260917")
    parser.add_argument("--model-name", default="stock-portfolio-swing-v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--max-assets", type=int, default=100)
    parser.add_argument("--train-end-date", default="20241231")
    parser.add_argument("--validation-start-date", default="20250101")
    parser.add_argument("--validation-end-date", default="20251231")
    parser.add_argument("--locked-start-date", default="20260101")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=120)
    parser.add_argument("--l2-regularization", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=0, help="0이면 validation에서 5/10/20/30 자동 선택")
    parser.add_argument("--initial-balance", type=float, default=10_000_000.0)
    parser.add_argument("--trading-fee", type=float, default=0.00015)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--trading-tax", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _selection_score(metrics: dict) -> float:
    # Validation에서 초과수익을 우선하되 과도한 낙폭과 낮은 CAGR을 약하게 억제한다.
    return float(
        metrics["excess_benchmark"]
        + 0.15 * metrics["cagr"]
        - 0.10 * metrics["max_drawdown"]
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        args.base_path, "output", "stock_portfolio_swing", timestamp
    )
    model_dir = os.path.join(args.base_path, "models", args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    frame = load_stock_portfolio_frame(
        args.base_path,
        args.dataset,
        codes=args.codes,
        max_assets=args.max_assets,
    )
    target, future_date = make_forward_target(frame.env, args.horizon)
    train_mask = (
        (frame.env["date"].to_numpy() <= str(args.train_end_date))
        & (future_date <= str(args.train_end_date))
        & np.isfinite(target)
    )
    print(f"제품: {args.model_name}")
    print(f"데이터셋: {args.dataset}, assets={len(frame.codes)}, dates={len(frame.dates)}, features={len(frame.feature_names)}")
    print(f"공통기간: {frame.dates[0]}~{frame.dates[-1]}")
    print(f"Forward target: {args.horizon} trading days, train labels={int(train_mask.sum()):,}")

    model = fit_forward_model(
        frame.features,
        target,
        train_mask,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
    )
    save_model(model, os.path.join(output_dir, "model_best.joblib"))
    save_model(model, os.path.join(output_dir, "model_final.joblib"))

    costs = {
        "initial_balance": args.initial_balance,
        "trading_fee": args.trading_fee,
        "slippage": args.slippage,
        "trading_tax": args.trading_tax,
    }
    top_k_candidates = [args.top_k] if args.top_k > 0 else [5, 10, 20, 30]
    candidates = []
    for top_k in top_k_candidates:
        metrics, _ = backtest_portfolio(
            frame.env,
            frame.features,
            frame.codes,
            model,
            start_date=args.validation_start_date,
            end_date=args.validation_end_date,
            top_k=top_k,
            **costs,
        )
        item = {"top_k": top_k, **metrics, "selection_score": _selection_score(metrics)}
        candidates.append(item)
        print(
            f"[validation top_k={top_k:2d}] return={metrics['total_return']*100:.2f}% "
            f"benchmark={metrics['benchmark_return']*100:.2f}% "
            f"excess={metrics['excess_benchmark']*100:.2f}% "
            f"sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']*100:.2f}%"
        )
    selected = max(candidates, key=lambda item: item["selection_score"])
    top_k = int(selected["top_k"])
    locked, _ = backtest_portfolio(
        frame.env,
        frame.features,
        frame.codes,
        model,
        start_date=args.locked_start_date,
        top_k=top_k,
        **costs,
    )
    print(
        f"[locked top_k={top_k:2d}] return={locked['total_return']*100:.2f}% "
        f"benchmark={locked['benchmark_return']*100:.2f}% "
        f"excess={locked['excess_benchmark']*100:.2f}% "
        f"sharpe={locked['sharpe']:.2f} mdd={locked['max_drawdown']*100:.2f}%"
    )

    config = {
        "model_name": args.model_name,
        "model_type": "stock-portfolio-swing-supervised",
        "product": "stock-portfolio-swing",
        "dataset": args.dataset,
        "stock_codes": frame.codes,
        "date_start": frame.dates[0],
        "date_end": frame.dates[-1],
        "train_end_date": args.train_end_date,
        "validation_start_date": args.validation_start_date,
        "validation_end_date": args.validation_end_date,
        "locked_evaluation_start_date": args.locked_start_date,
        "feature_names": frame.feature_names,
        "feature_dim": len(frame.feature_names),
        "target": {
            "type": "forward_log_close_return",
            "horizon_trading_days": args.horizon,
            "label_cutoff": args.train_end_date,
            "execution": "feature at t -> close-to-close portfolio return from t to t+1",
        },
        "model_params": {
            "estimator": "HistGradientBoostingRegressor",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_samples_leaf": args.min_samples_leaf,
            "l2_regularization": args.l2_regularization,
            "seed": args.seed,
        },
        "portfolio": {
            "allocation": "equal_weight_top_k",
            "top_k": top_k,
            "candidate_top_k": [item["top_k"] for item in candidates],
            "selection_metric": "validation_excess_benchmark + 0.15*cagr - 0.10*max_drawdown",
        },
        "cost_assumptions": costs,
        "train_rows": int(train_mask.sum()),
        "validation": {
            "selected": selected,
            "candidates": candidates,
        },
        "locked_evaluation": locked,
        "deployment_approved": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    manifest = {
        **config,
        "candidate_output_dir": os.path.abspath(output_dir),
        "checkpoints": ["model_best.joblib", "model_final.joblib"],
        "validation_artifact": os.path.abspath(os.path.join(output_dir, "train_config.json")),
        "deployment_approved": False,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    for filename in ("model_best.joblib", "model_final.joblib", "train_config.json", "manifest.json"):
        shutil.copy2(os.path.join(output_dir, filename), os.path.join(model_dir, filename))
    with open(os.path.join(model_dir, "universe.json"), "w", encoding="utf-8") as handle:
        json.dump({"codes": frame.codes, "dates": frame.dates, "feature_names": frame.feature_names}, handle, ensure_ascii=False, indent=2)
    dataset_dir = args.dataset if os.path.isabs(args.dataset) else os.path.join(args.base_path, "data", args.dataset)
    if os.path.exists(os.path.join(dataset_dir, "scaler.pkl")):
        shutil.copy2(os.path.join(dataset_dir, "scaler.pkl"), os.path.join(model_dir, "scaler.pkl"))
    if os.path.exists(os.path.join(dataset_dir, "dataset_meta.json")):
        shutil.copy2(os.path.join(dataset_dir, "dataset_meta.json"), os.path.join(model_dir, "dataset_meta.json"))
    print(f"학습 완료: {output_dir}")
    print(f"모델 저장: {model_dir}")


if __name__ == "__main__":
    main()
