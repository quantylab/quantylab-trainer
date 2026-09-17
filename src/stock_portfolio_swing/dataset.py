"""Causal, common-calendar loader for the stock portfolio swing product."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StockPortfolioFrame:
    env: pd.DataFrame
    features: np.ndarray
    feature_names: list[str]
    codes: list[str]
    dates: list[str]


def _dataset_dir(base_path: str, dataset: str) -> str:
    return dataset if os.path.isabs(dataset) else os.path.join(base_path, "data", dataset)


def load_stock_portfolio_frame(
    base_path: str,
    dataset: str,
    *,
    codes: list[str] | None = None,
    max_assets: int = 100,
    min_common_dates: int = 500,
) -> StockPortfolioFrame:
    """Load a fixed stock universe on one common trading calendar.

    The source snapshot contains recently listed stocks with shorter histories.
    A portfolio model cannot silently treat those missing periods as returns, so
    the default universe is selected by longest available history and then
    restricted to dates common to every selected stock.
    """

    directory = _dataset_dir(base_path, dataset)
    env_path = os.path.join(directory, "environment.csv")
    feature_path = os.path.join(directory, "training_scaled.csv")
    if not os.path.exists(env_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(f"stock portfolio dataset 파일이 없습니다: {directory}")

    env = pd.read_csv(
        env_path,
        dtype={"date": str, "stock_code": str},
        low_memory=False,
    )
    feature_frame = pd.read_csv(feature_path)
    if len(env) != len(feature_frame):
        raise ValueError(f"stock 데이터 정렬 불일치: env={len(env)}, features={len(feature_frame)}")

    env = env.copy()
    env["date"] = env["date"].astype(str).str.zfill(8)
    env["stock_code"] = env["stock_code"].astype(str).str.zfill(6)
    feature_values = np.nan_to_num(
        feature_frame.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    counts = env.groupby("stock_code", sort=True)["date"].size()
    if codes is None:
        if max_assets < 2:
            raise ValueError("max_assets는 2 이상이어야 합니다.")
        candidates = (
            counts.rename("rows")
            .rename_axis("stock_code")
            .reset_index()
            .sort_values(["rows", "stock_code"], ascending=[False, True])
        )
        selected_codes = sorted(candidates.head(max_assets)["stock_code"].tolist())
    else:
        selected_codes = sorted({str(code).zfill(6) for code in codes})
        missing = sorted(set(selected_codes) - set(counts.index))
        if missing:
            raise ValueError(f"dataset에 없는 stock code: {missing[:10]}")

    if len(selected_codes) < 2:
        raise ValueError("portfolio 학습에는 2개 이상의 stock이 필요합니다.")

    common_dates: set[str] | None = None
    for code in selected_codes:
        dates = set(env.loc[env["stock_code"].eq(code), "date"])
        common_dates = dates if common_dates is None else common_dates & dates
    dates = sorted(common_dates or set())
    if len(dates) < min_common_dates:
        raise ValueError(
            f"공통 거래일이 너무 적습니다: {len(dates)}일 < {min_common_dates}일 "
            f"(assets={len(selected_codes)})"
        )

    selected_mask = env["stock_code"].isin(selected_codes) & env["date"].isin(dates)
    selected_env = env.loc[selected_mask].copy()
    selected_features = feature_values[selected_mask.to_numpy()]
    order = np.lexsort(
        (selected_env["date"].to_numpy(), selected_env["stock_code"].to_numpy())
    )
    selected_env = selected_env.iloc[order].reset_index(drop=True)
    selected_features = selected_features[order]

    actual_codes = sorted(selected_env["stock_code"].unique().tolist())
    actual_dates = sorted(selected_env["date"].unique().tolist())
    if actual_codes != selected_codes or actual_dates != dates:
        raise ValueError("stock portfolio common-calendar 정렬에 실패했습니다.")

    return StockPortfolioFrame(
        env=selected_env,
        features=selected_features,
        feature_names=list(feature_frame.columns),
        codes=actual_codes,
        dates=actual_dates,
    )
