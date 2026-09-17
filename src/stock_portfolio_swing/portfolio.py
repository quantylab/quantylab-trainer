"""Forward-return training and long-only portfolio evaluation utilities."""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def make_forward_target(env: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a per-stock forward log-return target without look-ahead."""

    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    grouped = env.groupby("stock_code", sort=False)
    future_close = grouped["close"].shift(-horizon)
    future_date = grouped["date"].shift(-horizon).fillna("").astype(str).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.log(future_close.to_numpy(dtype=float) / env["close"].to_numpy(dtype=float))
    target = np.clip(target, -0.8, 0.8).astype(np.float32)
    target[~np.isfinite(target)] = np.nan
    return target, future_date


def fit_forward_model(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    max_iter: int = 220,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 120,
    l2_regularization: float = 2.0,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(target)
    if int(valid.sum()) < 1000:
        raise ValueError(f"학습 가능한 label이 너무 적습니다: {int(valid.sum())}")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    model.fit(features[valid], target[valid])
    return model


def _metrics(values: list[float], benchmark: list[float], turnover: list[float], initial_balance: float) -> dict:
    returns = np.asarray(values, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark, dtype=np.float64)
    if not len(returns):
        raise ValueError("평가 구간에 거래일이 없습니다.")
    wealth = np.cumprod(1.0 + returns)
    benchmark_wealth = np.cumprod(1.0 + benchmark_returns)
    years = len(returns) / 252.0
    cagr = float(wealth[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth[-1] > 0 else -1.0
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.max(1.0 - wealth / np.maximum(peak, 1e-12)))
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if returns.std() > 1e-12 else 0.0
    win_rate = float(np.mean(returns > 0))
    return {
        "total_return": float(wealth[-1] - 1.0),
        "benchmark_return": float(benchmark_wealth[-1] - 1.0),
        "excess_benchmark": float(wealth[-1] - benchmark_wealth[-1]),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "mean_daily_return": float(returns.mean()),
        "mean_turnover": float(np.mean(turnover)) if turnover else 0.0,
        "total_turnover": float(np.sum(turnover)) if turnover else 0.0,
        "num_days": int(len(returns)),
        "final_balance": float(initial_balance * wealth[-1]),
    }


def _frame_matrices(env: pd.DataFrame, features: np.ndarray, codes: list[str], model):
    price = env.pivot(index="date", columns="stock_code", values="close").reindex(columns=codes)
    if price.isna().any().any():
        raise ValueError("common-calendar price matrix에 결측이 있습니다.")
    predictions = model.predict(features).astype(np.float64)
    pred_frame = env[["date", "stock_code"]].copy()
    pred_frame["prediction"] = predictions
    pred = pred_frame.pivot(index="date", columns="stock_code", values="prediction").reindex(columns=codes)
    if pred.isna().any().any():
        raise ValueError("prediction matrix에 결측이 있습니다.")
    return price.sort_index(), pred.sort_index()


def backtest_portfolio(
    env: pd.DataFrame,
    features: np.ndarray,
    codes: list[str],
    model,
    *,
    start_date: str,
    end_date: str | None = None,
    top_k: int = 10,
    initial_balance: float = 10_000_000.0,
    trading_fee: float = 0.00015,
    slippage: float = 0.0003,
    trading_tax: float = 0.002,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate daily close-to-close rebalancing with costs and cash.

    Predictions at date ``t`` determine holdings for ``t -> t+1``.  This
    makes the execution boundary explicit and prevents using the next close
    when selecting the portfolio.
    """

    price, prediction = _frame_matrices(env, features, codes, model)
    date_index = price.index.astype(str)
    selected_dates = date_index[(date_index >= str(start_date))]
    if end_date is not None:
        selected_dates = selected_dates[selected_dates <= str(end_date)]
    if len(selected_dates) < 2:
        raise ValueError(f"평가 구간이 너무 짧습니다: {start_date}~{end_date}")

    top_k = min(max(int(top_k), 1), len(codes))
    code_index = {code: index for index, code in enumerate(codes)}
    previous_weights = np.zeros(len(codes), dtype=np.float64)
    values = []
    benchmark = []
    turnover = []
    rows = []
    for current_date, next_date in zip(selected_dates[:-1], selected_dates[1:]):
        scores = prediction.loc[current_date].to_numpy(dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        selected = order[:top_k]
        weights = np.zeros(len(codes), dtype=np.float64)
        weights[selected] = 1.0 / top_k

        daily_returns = (
            price.loc[next_date].to_numpy(dtype=np.float64)
            / price.loc[current_date].to_numpy(dtype=np.float64)
            - 1.0
        )
        sell_weights = np.maximum(previous_weights - weights, 0.0)
        day_turnover = float(np.abs(weights - previous_weights).sum())
        cost = day_turnover * (trading_fee + slippage) + float(sell_weights.sum()) * trading_tax
        portfolio_return = float(np.dot(weights, daily_returns) - cost)
        benchmark_return = float(np.mean(daily_returns))
        values.append(portfolio_return)
        benchmark.append(benchmark_return)
        turnover.append(day_turnover)
        rows.append({
            "date": str(current_date),
            "next_date": str(next_date),
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "cost": cost,
            "turnover": day_turnover,
            "selected_codes": [codes[index] for index in selected],
            "selected_weights": {codes[index]: float(weights[index]) for index in selected},
        })
        previous_weights = weights

    metrics = _metrics(values, benchmark, turnover, initial_balance)
    metrics.update({
        "period_start": str(selected_dates[0]),
        "period_end": str(selected_dates[-1]),
        "top_k": top_k,
        "asset_count": len(codes),
        "trading_fee": trading_fee,
        "slippage": slippage,
        "trading_tax": trading_tax,
    })
    return metrics, pd.DataFrame(rows)


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: str):
    return joblib.load(path)
