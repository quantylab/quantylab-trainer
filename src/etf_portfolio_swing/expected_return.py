"""Train a causal ETF expected-return estimator for prediction display.

The portfolio policy remains responsible for allocation.  This companion
regressor provides the per-ETF expected-return score shown by the daily
prediction API, so allocation weights are not mislabeled as returns.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _load_dataset(dataset_dir: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    env = pd.read_csv(os.path.join(dataset_dir, "environment.csv"), dtype={"etf_code": str})
    env["etf_code"] = env["etf_code"].str.zfill(6)
    env["date"] = env["date"].astype(str)
    features_df = pd.read_csv(os.path.join(dataset_dir, "training_scaled.csv"))
    if len(env) != len(features_df):
        raise ValueError(f"environment/features 행 수가 다릅니다: {len(env)} != {len(features_df)}")

    order = env.sort_values(["etf_code", "date"]).index
    env = env.loc[order].reset_index(drop=True)
    features = features_df.loc[order].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = [str(column) for column in features_df.columns]
    return env, features, feature_names


def _make_target(env: pd.DataFrame, target_kind: str, horizon: int) -> tuple[np.ndarray, dict]:
    if target_kind not in {"swing", "intraday"}:
        raise ValueError("target_kind는 swing 또는 intraday여야 합니다.")
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")

    if target_kind == "intraday":
        with np.errstate(divide="ignore", invalid="ignore"):
            target = (env["close"].to_numpy(dtype=float) / env["open"].to_numpy(dtype=float)) - 1.0
        metadata = {
            "type": "intraday_simple_return",
            "horizon": "same_trading_day",
            "execution": "session open -> session close",
        }
    else:
        future_close = env.groupby("etf_code", sort=False)["close"].shift(-horizon)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.log(future_close.to_numpy(dtype=float) / env["close"].to_numpy(dtype=float))
        metadata = {
            "type": "forward_log_close_return",
            "horizon_trading_days": horizon,
            "execution": "feature at t -> close at t+horizon",
        }
    target = np.clip(target, -0.8, 0.8)
    target[~np.isfinite(target)] = np.nan
    return target.astype(np.float32), metadata


def _metrics(prediction: np.ndarray, target: np.ndarray, target_type: str) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    if not len(target):
        return {"rows": 0}
    correlation = np.corrcoef(prediction, target)[0, 1] if len(target) > 1 else 0.0
    if target_type == "forward_log_close_return":
        target_pct = np.expm1(target) * 100.0
        prediction_pct = np.expm1(prediction) * 100.0
    else:
        target_pct = target * 100.0
        prediction_pct = prediction * 100.0
    return {
        "rows": int(len(target)),
        "rmse_log_return": float(math.sqrt(mean_squared_error(target, prediction))),
        "mae_log_return": float(mean_absolute_error(target, prediction)),
        "correlation": float(correlation) if np.isfinite(correlation) else 0.0,
        "directional_accuracy": float(np.mean((prediction > 0) == (target > 0))),
        "mean_target_return_pct": float(np.mean(target_pct)),
        "mean_predicted_return_pct": float(np.mean(prediction_pct)),
    }


def train_expected_return_model(
    dataset_dir: str,
    output_dir: str,
    *,
    target_kind: str = "swing",
    horizon: int = 20,
    train_end_date: str = "20241231",
    validation_end_date: str = "20251231",
    max_iter: int = 220,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 120,
    l2_regularization: float = 2.0,
) -> dict:
    env, features, feature_names = _load_dataset(dataset_dir)
    target, target_metadata = _make_target(env, target_kind, horizon)
    train_mask = (env["date"] <= train_end_date) & np.isfinite(target)
    validation_mask = (
        (env["date"] > train_end_date)
        & (env["date"] <= validation_end_date)
        & np.isfinite(target)
    )
    locked_mask = (env["date"] > validation_end_date) & np.isfinite(target)
    if int(train_mask.sum()) < 1000:
        raise ValueError(f"학습 가능한 ETF return label이 너무 적습니다: {int(train_mask.sum())}")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=42,
    )
    model.fit(features[train_mask], target[train_mask])
    predictions = model.predict(features)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "expected_return_model.joblib")
    metadata_path = os.path.join(output_dir, "expected_return_meta.json")
    joblib.dump(model, model_path)
    metadata = {
        "model_type": "etf-expected-return-supervised",
        "score_type": "expected_return",
        "score_unit": "percent",
        "dataset": os.path.abspath(dataset_dir),
        "target": target_metadata,
        "feature_names": feature_names,
        "feature_dim": int(features.shape[1]),
        "train_end_date": train_end_date,
        "validation_end_date": validation_end_date,
        "metrics": {
            "train": _metrics(predictions[train_mask], target[train_mask], target_metadata["type"]),
            "validation": _metrics(predictions[validation_mask], target[validation_mask], target_metadata["type"]),
            "locked": _metrics(predictions[locked_mask], target[locked_mask], target_metadata["type"]),
        },
        "deployment_approved": False,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF expected-return 모델 학습")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-kind", choices=["swing", "intraday"], default="swing")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--train-end-date", default="20241231")
    parser.add_argument("--validation-end-date", default="20251231")
    args = parser.parse_args()
    metadata = train_expected_return_model(
        args.dataset_dir,
        args.output_dir,
        target_kind=args.target_kind,
        horizon=args.horizon,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
    )
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {os.path.join(args.output_dir, 'expected_return_model.joblib')}")


if __name__ == "__main__":
    main()
