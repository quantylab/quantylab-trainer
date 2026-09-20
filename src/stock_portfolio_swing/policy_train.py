"""Train and evaluate the cash-aware stock portfolio policy.

This product deliberately uses the same PPO policy/environment contract as the
ETF portfolio product: an action is N stock weights plus one cash weight.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import torch

from .dataset import StockPortfolioFrame, load_stock_portfolio_frame
from ..etf_portfolio_swing.agent import PortfolioAgent
from ..etf_portfolio_swing.backtest import run_backtest
from ..etf_portfolio_swing.environment import PortfolioTradingEnvironment
from ..etf_portfolio_swing.network import PortfolioPolicyNetwork, PortfolioValueNetwork
from ..etf_portfolio_swing.trainer import PortfolioPPOTrainer


def _asset_inputs(frame: StockPortfolioFrame, end_date: str) -> tuple[dict, dict]:
    """Convert the common-calendar stock frame to the shared PPO environment."""
    env = frame.env.copy()
    env["date"] = env["date"].astype(str)
    mask = env["date"].le(end_date)
    env = env.loc[mask].reset_index(drop=True)
    feature_rows = frame.features[mask.to_numpy()]
    asset_data, asset_features = {}, {}
    for code in frame.codes:
        code_mask = env["stock_code"].eq(code).to_numpy()
        data = env.loc[code_mask, ["date", "close"]].copy()
        if len(data) < 60:
            raise ValueError(f"{code}: policy training history is too short")
        asset_data[code] = data
        asset_features[code] = feature_rows[code_mask]
    return asset_data, asset_features


def _env(frame: StockPortfolioFrame, end_date: str, *, oos_start: str | None = None, lookback: int = 20):
    data, features = _asset_inputs(frame, end_date)
    return PortfolioTradingEnvironment(
        asset_data=data, asset_features=features, lookback=lookback,
        initial_balance=10_000_000.0, trading_fee=0.00015,
        reward_scale=10.0, fee_penalty_scale=5.0,
        drawdown_penalty_threshold=0.10, drawdown_penalty_scale=15.0,
        rolling_sharpe_scale=2.0, reward_terminal_scale=30.0,
        oos_start_date=oos_start,
    )


def _jsonable(value):
    """Convert numpy scalar/container values produced by the environment."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    p = argparse.ArgumentParser(description="Cash-aware stock portfolio PPO training")
    p.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    p.add_argument("--dataset", default="stock_20260917")
    p.add_argument("--model-name", default="stock-portfolio-swing-v2")
    p.add_argument("--max-assets", type=int, default=100)
    p.add_argument("--train-end-date", default="20241231")
    p.add_argument("--validation-start-date", default="20250101")
    p.add_argument("--validation-end-date", default="20251231")
    p.add_argument("--locked-start-date", default="20260101")
    p.add_argument("--locked-end-date", default="20260916")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--update-interval", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-checkpoint", default=None, help="검증만 수행할 기존 policy_best.pt 경로")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    frame = load_stock_portfolio_frame(args.base_path, args.dataset, max_assets=args.max_assets)
    train_env = _env(frame, args.train_end_date, lookback=args.lookback)
    policy = PortfolioPolicyNetwork(train_env.n_assets, train_env.n_features, args.d_model, args.n_heads)
    value = PortfolioValueNetwork(train_env.n_assets, train_env.n_features, args.d_model, args.n_heads)
    agent = PortfolioAgent(policy, value, lr_policy=0.0002, lr_value=0.0005, device=args.device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.base_path, "output", "stock_portfolio_swing", run_id)
    log_dir = os.path.join(args.base_path, "logs", "stock_portfolio_swing", run_id)
    if args.resume_checkpoint:
        agent.load(args.resume_checkpoint)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(args.resume_checkpoint, os.path.join(output_dir, "policy_best.pt"))
        shutil.copy2(args.resume_checkpoint, os.path.join(output_dir, "policy_final.pt"))
    else:
        trainer = PortfolioPPOTrainer(
            train_env, agent, num_episodes=args.episodes, update_interval=args.update_interval,
            output_dir=output_dir, log_dir=log_dir,
        )
        trainer.train()

    # Select the best train checkpoint, then evaluate sequential validation and locked periods.
    agent.load(os.path.join(output_dir, "policy_best.pt"))
    validation, _ = run_backtest(_env(frame, args.validation_end_date, oos_start=args.validation_start_date, lookback=args.lookback), agent)
    locked, _ = run_backtest(_env(frame, args.locked_end_date, oos_start=args.locked_start_date, lookback=args.lookback), agent)
    validation = _jsonable(validation)
    locked = _jsonable(locked)
    print("validation", json.dumps(validation, ensure_ascii=False))
    print("locked", json.dumps(locked, ensure_ascii=False))

    config = {
        "model_name": args.model_name,
        "model_type": "stock-portfolio-swing-ppo-policy",
        "product": "stock-portfolio-swing",
        "dataset": args.dataset,
        "stock_codes": frame.codes,
        "n_assets": train_env.n_assets,
        "n_features": train_env.n_features,
        "feature_names": frame.feature_names,
        "lookback": args.lookback,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "policy_output": "asset target weights plus cash weight (last index)",
        "execution": "feature at market close -> next close-to-close portfolio return",
        "train_end_date": args.train_end_date,
        "validation_start_date": args.validation_start_date,
        "validation_end_date": args.validation_end_date,
        "locked_evaluation_start_date": args.locked_start_date,
        "locked_evaluation_end_date": args.locked_end_date,
        "cost_assumptions": {"trading_fee": 0.00015, "slippage": 0.0003, "trading_tax": 0.002},
        "validation": validation,
        "locked_evaluation": locked,
        "companion_expected_return_model": "stock-portfolio-swing-v1",
        "deployment_approved": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "candidate_output_dir": output_dir, "checkpoints": ["policy_best.pt", "policy_final.pt"]}, f, ensure_ascii=False, indent=2)

    model_dir = os.path.join(args.base_path, "models", args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    for name in ("policy_best.pt", "policy_final.pt", "train_config.json", "manifest.json"):
        shutil.copy2(os.path.join(output_dir, name), os.path.join(model_dir, name))
    dataset_dir = args.dataset if os.path.isabs(args.dataset) else os.path.join(args.base_path, "data", args.dataset)
    for name in ("scaler.pkl", "dataset_meta.json"):
        source = os.path.join(dataset_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(model_dir, name))
    print(f"candidate model saved: {model_dir}")


if __name__ == "__main__":
    main()
