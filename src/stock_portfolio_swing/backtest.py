"""Backtest a trained stock-portfolio-swing model."""

from __future__ import annotations

import argparse
import json
import os

from .dataset import load_stock_portfolio_frame
from .portfolio import backtest_portfolio, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock portfolio swing 백테스트")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default="stock-portfolio-swing-v1")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model if os.path.isabs(args.model) else os.path.join(args.base_path, "models", args.model)
    with open(os.path.join(model_dir, "train_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    dataset = args.dataset or config["dataset"]
    frame = load_stock_portfolio_frame(
        args.base_path,
        dataset,
        codes=config["stock_codes"],
        max_assets=len(config["stock_codes"]),
    )
    model = load_model(os.path.join(model_dir, "model_best.joblib"))
    start_date = args.start_date or config["locked_evaluation_start_date"]
    top_k = args.top_k or int(config["portfolio"]["top_k"])
    metrics, daily = backtest_portfolio(
        frame.env,
        frame.features,
        frame.codes,
        model,
        start_date=start_date,
        end_date=args.end_date,
        top_k=top_k,
        **config["cost_assumptions"],
    )
    output = args.output or os.path.join(model_dir, "locked_backtest_result.json")
    result = {
        "model_name": config["model_name"],
        "model_type": config["model_type"],
        "dataset": dataset,
        "period": {"start_date": start_date, "end_date": args.end_date},
        "portfolio": {"top_k": top_k, "asset_count": len(frame.codes)},
        "cost_assumptions": config["cost_assumptions"],
        "aggregate": metrics,
        "daily": daily.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"결과 저장: {output}")


if __name__ == "__main__":
    main()

